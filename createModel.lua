local nninit = require 'nninit'

function getSoftMaxLayer()

    local model = nn.Sequential()

    model:add(nn.Reshape(L, k)) -- TODO: Change 120 to batchSize or inputSize
    model:add(nn.SplitTable(2))

    map = nn.MapTable()
    map:add(nn.SoftMax())

    model:add(map)

    model:add(nn.JoinTable(2))

    return model
end

function getImageModel()

    local model = loadcaffe.load(filePath .. 'CNN Model/trainnet.prototxt', filePath .. 'CNN Model/snapshot_iter_16000.caffemodel', 'cudnn')

    -- Remove classification layer and add hash layer
    model.modules[#model.modules] = nil
    model:add(nn.Linear(4096, hashLayerSize)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer))

    model:add(getSoftMaxLayer())

    return model
end

function getTextModel()

    local model = loadcaffe.load(filePath .. 'text model/tag_trainnet.prototxt', filePath .. 'text model/snapshot_iter_200.caffemodel', 'cudnn')

    -- Remove first layer
    model.modules[1] = nil
    for i = 1,#model.modules-1 do
        model.modules[i] = model.modules[i+1]
    end
    model.modules[#model.modules] = nil

    -- Remove classification layer and add hash layer
    model.modules[#model.modules] = nil

    -- State of Text Model at this point
    -- nn.Linear(1075, 1075)
    -- cudnn.ReLU(true))
    -- nn.Dropout(0.500000)
    -- nn.Linear(1075, 2048)
    -- cudnn.ReLU(true)
    -- nn.Dropout(0.500000)

    model:add(nn.Linear(2048, hashLayerSize)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer))

    model:add(getSoftMaxLayer())

    return model
end

function createCombinedModel(imageModel, textModel)

    local model = nn.Sequential()

    cnn_text = nn.ParallelTable()
    cnn_text:add(imageModel)
    cnn_text:add(textModel)

    model:add(cnn_text)
    model:add(nn.DotProduct())

    model = model:cuda()

    return model
end


function getCriterion()

    criterion = nn.MSECriterion()
    criterion.sizeAverage = false
    criterion = criterion:cuda()
    return criterion
end