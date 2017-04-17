--TODO: Rename model to imageClassifier and remove calcClassAccuracy functions
--TODO: Remove data loader methods and use aux methods

require 'nn'
require 'loadcaffe' -- doesn't work on server
require 'image'
require 'optim'
nninit = require 'nninit'

GPU = true
if (GPU) then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
end

matio = require 'matio'

dataPath = '/home/kjoslyn/torch/test/data/mirflickr/' -- server
-- dataPath = '../../Datasets/mirflickr/' -- labcomp
infoPath = '/home/kjoslyn/kevin/' -- server
-- infoPath = '../../kevin/' -- labcomp

function calcClassAccuracyOnTrainset(classifier)

    roundedOutput = calcRoundedOutputInBatches(classifier, torch.CudaTensor(trainset.data:size(1), 24), trainset.data)
    dotProd = torch.CudaTensor(Ntrain)
    for i = 1, Ntrain do
        dotProd[i] = torch.dot(roundedOutput[i]:float(), trainset.label[i])
    end
    zero = torch.zeros(Ntrain):cuda()
    numCorrect = dotProd:gt(zero):sum()
    accuracy = numCorrect * 100 / Ntrain
    return accuracy
end

function calcRoundedOutputInBatches(classifier, output, data) 

    N = data:size(1)
    local batchSize = 128
    local numBatches = torch.ceil(N / batchSize)
    for b = 0, numBatches - 1 do
        startIndex = b * batchSize + 1
        endIndex = math.min((b + 1) * batchSize, N)
        batch = data[{{ startIndex, endIndex }}]
        output[{{ startIndex, endIndex}}] = classifier:cuda():forward(batch:cuda()):round()
    end
    return output    
end

function getBatch(batchNum, batchSize, perm)

    startIndex = batchNum * batchSize + 1
    endIndex = math.min((batchNum + 1) * batchSize, trainset:size())

    batchPerm = perm[ {{ startIndex, endIndex }} ]
    batch = {}
    batch.data = trainset.data:index(1, batchPerm)
    batch.label = trainset.label:index(1, batchPerm)
    batch.data = batch.data:cuda()
    batch.label = batch.label:cuda()

    setmetatable(batch, 
        {__index = function(t, i) 
                        return {t.data[i], t.label[i]} 
                    end}
    );

    function batch:size() 
        return self.data:size(1) 
    end

    return batch
end

function loadData(small)

    batchSize = 128

    print('Loading training images')
    if small then
        print('**** Warning: small = true')
        trainset = torch.load('mirflickr_trainset_small.t7')
    else
        trainset = torch.load('mirflickr_trainset.t7')
    end
    Ntrain = trainset.data:size(1)

    print('Loading test images')
    if small then
        testset = torch.load('mirflickr_testset_small.t7')
    else
        testset = torch.load('mirflickr_testset.t7')
    end
    Ntest = testset.data:size(1)

    setmetatable(trainset, 
        {__index = function(t, i) 
                        return {t.data[i], t.label[i]} 
                    end}
    );

    function trainset:size() 
        return self.data:size(1) 
    end

    mean = {} -- store the mean, to normalize the test set in the future
    stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

        stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
        testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    -- trainset, testset = loadImageData(small)
end

function loadModel()
    -- caffemodel = loadcaffe.load('trainnet.prototxt', 'snapshot_iter_16000.caffemodel', 'cudnn')
    model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3,96, 11, 11, 4, 4, 0, 0, 1):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, 1):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(nn.View(-1):setNumInputDims(3))
    model:add(nn.Linear(9216, 4096):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))
    model:add(nn.Linear(4096, 4096):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))
    model:add(nn.Linear(4096, 24):init('weight', nninit.xavier, {dist = 'normal', gain = 'sigmoid'}))

    model:add(nn.Sigmoid())
end

function trainAndEvaluate(numEpochs, startEpoch)
    -- criterion = nn.MultiLabelSoftMarginCriterion()
    -- criterion = nn.MSECriterion()
    criterion = nn.BCECriterion()

    if (GPU) then
        criterion = criterion:cuda()
        testset.data = testset.data:cuda()
        testset.label = testset.label:cuda()
        model:cuda()
    end

    params, gradParams = model:getParameters()

    local optimState = {
        learningRate = .01
        -- learningRateDecay = 1e-7
        -- learningRate = 1e-3,
        -- learningRateDecay = 1e-4,
        -- weightDecay = 0.01
        -- momentum = 0.9
    }

    batchSize = 128

    numBatches = math.ceil(Ntrain / batchSize)

    model:training()

    if not startEpoch then
        startEpoch = 1
    end
    for epoch = startEpoch, numEpochs do

        -- shuffle at each epoch
        perm = torch.randperm(Ntrain):long()
        totalLoss = 0
        totNumIncorrect = 0

        for batchNum = 0, numBatches - 1 do

            trainBatch = getBatch(batchNum, batchSize, perm)

            function feval(x)
                -- get new parameters
                if x ~= params then
                    params:copy(x)
                end

                gradParams:zero()

                local outputs = model:forward(trainBatch.data)
                local loss = criterion:forward(outputs, trainBatch.label)

                totalLoss = totalLoss + loss
                local roundedOutput = torch.CudaTensor(outputs:size(1), 24):copy(outputs):round()
                totNumIncorrect = totNumIncorrect + torch.ne(roundedOutput, trainBatch.label):sum()
                -- local numOnes = roundedOutput:sum()

                local dloss_doutputs = criterion:backward(outputs, trainBatch.label)
                model:backward(trainBatch.data, dloss_doutputs)

                return loss, gradParams
            end
            optim.sgd(feval, params, optimState)

            -- collectgarbage()
        end

        model:evaluate()
        avgLoss = totalLoss / numBatches
        print("Epoch " .. epoch .. ": avg loss = " .. avgLoss)
        avgNumIncorrect = totNumIncorrect / Ntrain
        print("Epoch " .. epoch .. ": avg num incorrect = " .. avgNumIncorrect)
        classAcc = calcClassAccuracyOnTrainset(model)
        print(string.format("Epoch %d: Trainset class accuracy = %.2f\n", epoch, classAcc))
        model:training()
    end
end