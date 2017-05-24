-- //////////////////////////////////////////
-- Typical flow:
-- require 'unimodal'
-- loadPackagesAndModel(datasetType, modality) -- 'mir' or 'nus', 'I', or 'X'
-- loadData() -- uses dataLoader.lua
-- optional: loadModelSnapshot -- from createModel.lua
-- trainAndEvaluate()
-- /////////////////////////////////////////

function loadPackagesAndModel(datasetType, modality)

    require 'nn'
    require 'optim'
    nninit = require 'nninit'
    gnuplot = require 'gnuplot'
    require 'auxf.evaluate'
    require 'auxf.dataLoader'
    require 'auxf.createModel'
    require 'imagenetloader.dataset'
    require 'cutorch'
    require 'cunn'
    require 'cudnn'

    p = {} -- parameters
    d = {} -- data
    m = {} -- models
    g = {} -- other global variables
    o = {} -- optimStates and model parameters

    matio = require 'matio'

    local snapshotDatasetDir
    if datasetType == 'mir' then
        p.numClasses = 24
        p.tagDim = 1075
        snapshotDatasetDir = '/mirflickr'
        g.datasetPath = '/home/kjoslyn/datasets/mirflickr/'
        g.plotNumEpochs = 5;
    elseif datasetType == 'nus' then
        p.numClasses = 21
        p.tagDim = 1000
        snapshotDatasetDir = '/nuswide'
        g.datasetPath = '/home/kjoslyn/datasets/nuswide/'
        g.plotNumEpochs = 1;
    else
        print("Error: Unrecognized datasetType!! Should be mir or nus")
    end
    p.datasetType = datasetType
    p.modality = modality

    g.snapshotDir = '/home/kjoslyn/kevin/Project/snapshots' .. snapshotDatasetDir

    g.numEpochsCompleted = 0

    -- //////////// Load image / text model
    if modality == 'I' then
       m.classifier = getImageModel()
       m.imageClassifier = m.classifier
       g.evalTrainAccEpochs = 1
    elseif modality == 'X' then
       m.classifier = getUntrainedTextModel()
       m.textClassifier = m.classifier
       g.evalTrainAccEpochs = 10
    else
       print('Error in unimodal.lua: Unrecognized modality')
    end

    g.accIdx = 0
    g.pastNAcc = torch.Tensor(g.plotNumEpochs)
    g.avgDataAcc = torch.Tensor()
    g.maxDataAcc = torch.Tensor()
    g.minDataAcc = torch.Tensor()
    g.pastNLoss = torch.Tensor(g.plotNumEpochs)
    g.avgDataLoss = torch.Tensor()
    g.maxDataLoss = torch.Tensor()
    g.minDataLoss = torch.Tensor()
end

function doGetBatchText(startIndex, endIndex, perm)

    local batchPerm = perm[ {{ startIndex, endIndex }} ]
    local batch = {}
    batch.data = d.trainset.data:index(1, batchPerm)
    batch.label = d.trainset.label:index(1, batchPerm)
    batch.data = batch.data:cuda()
    batch.label = batch.label:cuda()

    return batch
end

function doGetBatchImage(startIndex, endIndex, perm)

    local batch = {}
    -- TODO: Uncomment!!
    batch.data, batch.label = d.dataset:getBySplit({'training', 'pretraining'}, 'I', startIndex, endIndex, perm)
    -- batch.data, batch.label = d.dataset:getBySplit({'training'}, 'I', startIndex, endIndex, perm)
    batch.data = batch.data:cuda()
    batch.label = batch.label:cuda()

    return batch
end

function getBatch(batchNum, batchSize, perm)

    local startIndex = batchNum * batchSize + 1
    local endIndex = math.min((batchNum + 1) * batchSize, Ntrain)

    local batch
    if p.modality == 'I' then
        batch = doGetBatchImage(startIndex, endIndex, perm)
    else
        batch = doGetBatchText(startIndex, endIndex, perm)
    end

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

function loadData(useKFold, small)

    local imageRootPath = g.datasetPath .. 'ImageData'
    d.dataset = imageLoader{path=imageRootPath, sampleSize={3,227,227}, splitFolders={'training', 'pretraining', 'val', 'query'}}

    Ntrain = d.dataset:sizeTrain() + d.dataset:sizePretraining()
    -- Ntrain = d.dataset:sizeTrain()
    Ntest = d.dataset:sizeTest()
    Nval = d.dataset:sizeVal()

    d.trainset = {}
    d.valset = {}
    d.testset = {}
    if p.modality == 'X' then
        -- This will actually be used in training, and in evaluating the trainset accuracy
        d.trainset.data, d.trainset.label = d.dataset:getBySplit({'training', 'pretraining'}, 'X', 1, Ntrain)
        -- Ntrain = d.dataset:sizeTrain()
        -- d.trainset.data, d.trainset.label = d.dataset:getBySplit({'training'}, 'X', 1, Ntrain)
    else
        -- This will only be used in evaluating the trainset accuracy. Training will include pretraining
        -- The entire trainset (including pretrain) is too larget to hold in memory
        d.trainset.data, d.trainset.label = d.dataset:getBySplit('training', 'I', 1, d.dataset:sizeTrain())
    end
    d.valset.data, d.valset.label = d.dataset:getBySplit('val', p.modality, 1, Nval)
    d.testset.data, d.testset.label = d.dataset:getBySplit('query', p.modality, 1, Ntest)

    collectgarbage()
end

function trainAndEvaluate(batchSize, learningRate, momentum, weightDecay, numEpochs, printTrainsetAcc)

    local startEpoch = g.numEpochsCompleted + 1

    if not g.plotStartEpoch then
        g.plotStartEpoch = startEpoch
    end

    -- criterion = nn.MultiLabelSoftMarginCriterion()
    -- criterion = nn.MSECriterion()
    criterion = nn.BCECriterion()

    criterion = criterion:cuda()
    m.classifier:cuda()

    params, gradParams = m.classifier:getParameters()

    local optimState = {
        learningRate = learningRate, -- .01 works for mirflickr
        -- learningRateDecay = 1e-7
        -- learningRate = 1e-3,
        -- learningRateDecay = 1e-4,
        weightDecay = weightDecay, -- .01?
        momentum = momentum -- 0.9?
    }

    numBatches = math.ceil(Ntrain / batchSize)

    m.classifier:training()

    local epochTimer = torch.Timer()

    for epoch = startEpoch, startEpoch + numEpochs - 1 do

        epochTimer:reset()
        epochTimer:resume()
        -- shuffle at each epoch
        perm = torch.randperm(Ntrain):long()
        totalLoss = 0
        totNumIncorrect = 0

        for batchNum = 0, numBatches - 1 do

            trainBatch = getBatch(batchNum, batchSize, perm)
            collectgarbage()

            function feval(x)
                -- get new parameters
                if x ~= params then
                    params:copy(x)
                end

                gradParams:zero()

                outputs = m.classifier:forward(trainBatch.data)
                local loss = criterion:forward(outputs, trainBatch.label)

                totalLoss = totalLoss + loss
                local roundedOutput = torch.CudaTensor(outputs:size(1), p.numClasses):copy(outputs):round()
                totNumIncorrect = totNumIncorrect + torch.ne(roundedOutput, trainBatch.label):sum()
                -- local numOnes = roundedOutput:sum()

                local dloss_doutputs = criterion:backward(outputs, trainBatch.label)
                m.classifier:backward(trainBatch.data, dloss_doutputs)

                return loss, gradParams
            end
            optim.sgd(feval, params, optimState)

            -- collectgarbage()
        end
        epochTimer:stop()

        collectgarbage()
        m.classifier:evaluate()
        avgLoss = totalLoss / numBatches
        print("Epoch " .. epoch .. ": avg loss = " .. avgLoss)
        avgNumIncorrect = totNumIncorrect / Ntrain
        print("Epoch " .. epoch .. ": avg num incorrect = " .. avgNumIncorrect)
        local valClassAcc
        if d.valset then
            valClassAcc = getClassAccuracy(d.valset.data, d.valset.label:cuda())
            print(string.format("Epoch %d: Val set class accuracy = %.2f", epoch, valClassAcc))
        end
        if d.trainset and epoch % g.evalTrainAccEpochs == 0 then
            local trainClassAcc = getClassAccuracy(d.trainset.data, d.trainset.label:cuda())
            print(string.format("Epoch %d: Train set class accuracy = %.2f", epoch, trainClassAcc))
        end
        print(string.format('Epoch time: %.2f seconds\n', epochTimer:time().real))
        m.classifier:training()

        g.accIdx = g.accIdx + 1
        g.pastNAcc[g.accIdx] = valClassAcc
        g.pastNLoss[g.accIdx] = math.min(avgLoss * 100, 100) -- Scale loss to fit in the same y axes as accuracy
        if g.accIdx % g.plotNumEpochs == 0 then
            g.avgDataAcc = g.avgDataAcc:cat(torch.Tensor({g.pastNAcc:mean()}))
            g.maxDataAcc = g.maxDataAcc:cat(torch.Tensor({g.pastNAcc:max()}))
            g.minDataAcc = g.minDataAcc:cat(torch.Tensor({g.pastNAcc:min()}))
            g.avgDataLoss = g.avgDataLoss:cat(torch.Tensor({g.pastNLoss:mean()}))
            g.maxDataLoss = g.maxDataLoss:cat(torch.Tensor({g.pastNLoss:max()}))
            g.minDataLoss = g.minDataLoss:cat(torch.Tensor({g.pastNLoss:min()}))
            plotClassAccAndLoss(epoch, true)
            g.accIdx = 0
        end

        local saveSnapshot
        if p.modality == 'I' then
            saveSnapshot = epoch % 10 == 0
        else
            saveSnapshot = epoch == 500 or epoch == 750 or epoch == 900 or epoch == 1000
        end

        if saveSnapshot then
        -- if epoch == 300 or epoch == 500 or epoch == 750 or epoch == 900 or epoch == 1000 then
            local paramsToSave, gp = m.classifier:getParameters()
            local date = os.date("*t", os.time())
            local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
            local modalityDir
            if p.modality == 'I' then
                modalityDir = 'imageNet'
            else
                modalityDir = 'textNet'
            end
            local snapshotFile = g.snapshotDir .. "/" .. modalityDir .. "/" .. dateStr .. "_snapshot_epoch_" .. epoch .. ".t7" 
            local snapshot = {}
            snapshot.params = paramsToSave
            snapshot.gparams = gp
            torch.save(snapshotFile, snapshot)
        end

        g.numEpochsCompleted = g.numEpochsCompleted + 1
    end
end