require 'nn'
require 'cunn'
require 'cudnn'
require 'dpnn'

torch. manualSeed(123)
local emb_learning_rate = 0.001
local vocab_size = 20
local emb_size = 10
local start_l = 1
local end_l = 6
local start_r = 11
local end_r = 19

local ll = end_l - start_l + 1
local lr = end_r - start_r + 1
local emb = nn.LookupTable(vocab_size,emb_size)



for i = 1, 200 do	
	lstart = (i)%15 +1
	lend = (i)%15 + 5
	rstart = (i)%11 +1
	rend = (i)%11 +8
	local lsent = torch.range(lstart,lend)
	local rsent = torch.range(rstart,rend)
	emb:forward(lsent)
	linput = torch.Tensor(emb.output:size()):copy(emb.output)
	rinput = emb:forward(rsent)


	linput_grads = torch.randn(ll,emb_size)
	rinput_grads = torch.randn(lr, emb_size)


	print('gradients before: ' .. emb.gradWeight:sum())

	emb:backward(lsent, linput_grads)
	emb:backward(rsent,rinput_grads)

	print('gradients after: ' .. emb.gradWeight:sum())

	print('weights before: ' .. emb.weight:sum())

	emb:updateParameters(emb_learning_rate)

	print('weights after: ' .. emb.weight:sum())
end