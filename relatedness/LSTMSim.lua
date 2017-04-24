--[[

  Semantic relatedness prediction using LSTMs.

--]]

local LSTMSim = torch.class('treelstm.LSTMSim')

function LSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
  self.sim_nhidden   = config.sim_nhidden   or 50
  self.update_emb = config.update_emb or 'false'
  self.emb_learning_rate = config.emb_learning_rate or 0.05
  self.debug_file = config.debug_file
  

  if self.update_emb == 'true' then
    self.update_emb = true
  else
    self.update_emb = false
  end
  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)

  if self.update_emb == true then
    self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
    self.emb:cuda()
    self.emb.weight:copy(config.emb_vecs:cuda())
  else
    self.emb_vecs = config.emb_vecs
  end

  -- number of similarity rating classes
  self.num_classes = 2

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  self.criterion = nn.ModuleCriterion(nn.ClassNLLCriterion(), nn.Log())
  self.criterion:cuda()
  
  -- initialize LSTM model
  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = false,
  }

  if self.structure == 'lstm' then
    self.llstm = treelstm.LSTM(lstm_config) -- "left" LSTM
    self.rlstm = treelstm.LSTM(lstm_config) -- "right" LSTM
  elseif self.structure == 'bilstm' then
    self.llstm = treelstm.LSTM(lstm_config)
    self.llstm_b = treelstm.LSTM(lstm_config) -- backward "left" LSTM
    self.rlstm = treelstm.LSTM(lstm_config)
    self.rlstm_b = treelstm.LSTM(lstm_config) -- backward "right" LSTM
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  -- similarity model
  self.sim_module = self:new_sim_module()
  local modules = nn.Parallel()
    :add(self.llstm)
    :add(self.sim_module)
  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  share_params(self.rlstm, self.llstm)
  if self.structure == 'bilstm' then
    -- tying the forward and backward weights improves performance
    share_params(self.llstm_b, self.llstm)
    share_params(self.rlstm_b, self.llstm)
  end
end

function LSTMSim:new_sim_module()
  local lvec, rvec, inputs, input_dim
  if self.structure == 'lstm' then
    -- standard (left-to-right) LSTM
    input_dim = 2 * self.num_layers * self.mem_dim
    local linput, rinput = nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec, rvec = linput, rinput
    else
      lvec, rvec = nn.JoinTable(1)(linput), nn.JoinTable(1)(rinput)
    end
    inputs = {linput, rinput}
  elseif self.structure == 'bilstm' then
    -- bidirectional LSTM
    input_dim = 4 * self.num_layers * self.mem_dim
    local lf, lb, rf, rb = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec = nn.JoinTable(1){lf, lb}
      rvec = nn.JoinTable(1){rf, rb}
    else
      -- in the multilayer case, each input is a table of hidden vectors (one for each layer)
      lvec = nn.JoinTable(1){nn.JoinTable(1)(lf), nn.JoinTable(1)(lb)}
      rvec = nn.JoinTable(1){nn.JoinTable(1)(rf), nn.JoinTable(1)(rb)}
    end
    inputs = {lf, lb, rf, rb}
  end
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  local vecs_to_input = nn.gModule(inputs, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(input_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.SoftMax())
  sim_module:cuda()
  return sim_module
end

function LSTMSim:train(dataset)
  self.llstm:training()
  self.rlstm:training()
  if self.structure == 'bilstm' then
    self.llstm_b:training()
    self.rlstm_b:training()
  end

  local indices = torch.randperm(dataset.size)
  local zeros = torch.CudaTensor(self.mem_dim):zero()
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

   -- get target distributions for batch
    local targets = torch.CudaTensor(batch_size, 1):zero()
    for j = 1, batch_size do
      local idx = indices[i + j - 1]
      targets[j] = dataset.labels[idx]
    end

    local feval = function(x)
      self.grad_params:zero()
      if self.update_emb == true then
        self.emb:zeroGradParameters()
      end
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local linputs = nil
        local rinputs = nil
        if self.update_emb == true then
          self.emb:forward(lsent)
          linputs = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
          rinputs = self.emb:forward(rsent)
        else
          linputs = self.emb_vecs:index(1, lsent:long()):cuda()
          rinputs = self.emb_vecs:index(1, rsent:long()):cuda()
        end
        -- get sentence representations
        local inputs
        if self.structure == 'lstm' then
          inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
        elseif self.structure == 'bilstm' then
          inputs = {
            self.llstm:forward(linputs),
            self.llstm_b:forward(linputs, true), -- true => reverse
            self.rlstm:forward(rinputs),
            self.rlstm_b:forward(rinputs, true)
          }
        end

        -- compute relatedness
        local output = self.sim_module:forward(inputs)

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])
        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        local rep_grad = self.sim_module:backward(inputs, sim_grad)
        local linput_grads = nil
        local rinput_grads = nil
        if self.structure == 'lstm' then
          linput_grads, rinput_grads  = self:LSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
        elseif self.structure == 'bilstm' then
          self:BiLSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
        end
        
        
        
        if self.update_emb == true then
          self.emb:backward(rsent, rinput_grads)
          self.emb:forward(lsent)
          self.emb:backward(lsent, linput_grads)
        end
      end
      loss = loss / batch_size
      self.grad_params:div(batch_size)
      if self.update_emb == true then   
        self.emb.gradWeight:div(batch_size)
        self.emb:updateParameters(self.emb_learning_rate)
      end
      self.debug_file:writeString(i .. ', ' .. self.emb.gradWeight:norm() .. ', ' .. self.emb.weight:norm() .. '\n')
     
      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function LSTMSim:LSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
  local lgrad, rgrad
  if self.num_layers == 1 then
    lgrad = torch.CudaTensor(lsent:nElement(), self.mem_dim):zero()
    rgrad = torch.CudaTensor(rsent:nElement(), self.mem_dim):zero()
    lgrad[lsent:nElement()] = rep_grad[1]
    rgrad[rsent:nElement()] = rep_grad[2]
  else
    lgrad = torch.CudaTensor(lsent:nElement(), self.num_layers, self.mem_dim):zero()
    rgrad = torch.CudaTensor(rsent:nElement(), self.num_layers, self.mem_dim):zero()
    for l = 1, self.num_layers do
      lgrad[{lsent:nElement(), l, {}}] = rep_grad[1][l]
      rgrad[{rsent:nElement(), l, {}}] = rep_grad[2][l]
    end
  end
  local linput_grads = self.llstm:backward(linputs, lgrad)
  local rinput_grads = self.rlstm:backward(rinputs, rgrad)
  return linput_grads, rinput_grads
end

-- Bidirectional LSTM backward propagation
function LSTMSim:BiLSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
  local lgrad, lgrad_b, rgrad, rgrad_b
  if self.num_layers == 1 then
    lgrad   = torch.CudaTensor(lsent:nElement(), self.mem_dim):zero()
    lgrad_b = torch.CudaTensor(lsent:nElement(), self.mem_dim):zero()
    rgrad   = torch.CudaTensor(rsent:nElement(), self.mem_dim):zero()
    rgrad_b = torch.CudaTensor(rsent:nElement(), self.mem_dim):zero()
    lgrad[lsent:nElement()] = rep_grad[1]
    rgrad[rsent:nElement()] = rep_grad[3]
    lgrad_b[1] = rep_grad[2]
    rgrad_b[1] = rep_grad[4]
  else
    lgrad   = torch.CudaTensor(lsent:nElement(), self.num_layers, self.mem_dim):zero()
    lgrad_b = torch.CudaTensor(lsent:nElement(), self.num_layers, self.mem_dim):zero()
    rgrad   = torch.CudaTensor(rsent:nElement(), self.num_layers, self.mem_dim):zero()
    rgrad_b = torch.CudaTensor(rsent:nElement(), self.num_layers, self.mem_dim):zero()
    for l = 1, self.num_layers do
      lgrad[{lsent:nElement(), l, {}}] = rep_grad[1][l]
      rgrad[{rsent:nElement(), l, {}}] = rep_grad[3][l]
      lgrad_b[{1, l, {}}] = rep_grad[2][l]
      rgrad_b[{1, l, {}}] = rep_grad[4][l]
    end
  end
  self.llstm:backward(linputs, lgrad)
  self.llstm_b:backward(linputs, lgrad_b, true)
  self.rlstm:backward(rinputs, rgrad)
  self.rlstm_b:backward(rinputs, rgrad_b, true)
end

-- Predict the similarity of a sentence pair.
function LSTMSim:predict(lsent, rsent)
  self.llstm:evaluate()
  self.rlstm:evaluate()
  local linputs = nil
  local rinputs = nil
  local lrep = nil
  local rrep = nil
  if self.update_emb == true then
    self.emb:forward(lsent)
    linputs = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    lrep = self.llstm:forward(linputs)
    rinputs = self.emb:forward(rsent)
    rrep = self.rlstm:forward(rinputs)
    print(linputs:norm())
    print(rinputs:norm())
    os.exit(2)
  else
    linputs = self.emb_vecs:index(1, lsent:long()):cuda()
    lrep = self.llstm:forward(linputs)
    rinputs = self.emb_vecs:index(1, rsent:long()):cuda()
    rrep = self.rlstm:forward(rinputs)
  end
  local inputs
  if self.structure == 'lstm' then
    inputs = {lrep, rrep}
  elseif self.structure == 'bilstm' then
    self.llstm_b:evaluate()
    self.rlstm_b:evaluate()
    inputs = {
      self.llstm:forward(linputs),
      self.llstm_b:forward(linputs, true),
      self.rlstm:forward(rinputs),
      self.rlstm_b:forward(rinputs, true)
    }
  end
  local output = self.sim_module:forward(inputs)
  self.llstm:forget()
  self.rlstm:forget()
  if self.structure == 'bilstm' then
    self.llstm_b:forget()
    self.rlstm_b:forget()
  end
  return output
end

-- Produce similarity predictions for each sentence pair in the dataset.
function LSTMSim:predict_dataset(dataset,confusion, is_blind_set)
  local predictions = torch.CudaTensor(dataset.size,2)
  local pred_loss =0
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(lsent, rsent)
    if is_blind_set == false then
      local sim = dataset.labels[i]
      local pred_instance_loss = self.criterion:forward(predictions[i],sim)
      pred_loss = pred_loss + pred_instance_loss
      if confusion ~= nil then
        confusion:add(predictions[i],sim)
      end
    end
  end
  return {predictions, pred_loss/dataset.size}
end

function LSTMSim:print_config()
  local num_params = self.params:nElement()
  local num_sim_params = self:new_sim_module():getParameters():nElement()
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %s\n',   'udpate word vector', self.update_emb)
  printf('%-25s = %.2e\n',   'word vector update rate', self.emb_learning_rate)
  printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %s\n',   'LSTM structure', self.structure)
  printf('%-25s = %d\n',   'LSTM layers', self.num_layers)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end

--
-- Serialization
--

function LSTMSim:save(path)
  local embeddings = nil
  if self.update_emb == true then
    embeddings = self.emb.weight:float()
  else
    embeddings = self.emb_vecs:float()
  end
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = embeddings,
    learning_rate = self.learning_rate,
    num_layers    = self.num_layers,
    mem_dim       = self.mem_dim,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function LSTMSim.load(path)
  local state = torch.load(path)
  local model = treelstm.LSTMSim.new(state.config)
  model.params:copy(state.params)
  return model
end
