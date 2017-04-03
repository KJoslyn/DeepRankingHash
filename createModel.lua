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

    return createClassifierAndHasher(model, 4096)
end

function getTextModel()

    local model = loadcaffe.load(filePath .. 'text model/tag_trainnet.prototxt', filePath .. 'text model/snapshot_iter_200.caffemodel', 'cudnn')

    -- Remove first layer that comes from caffemodel
    model.modules[1] = nil
    for i = 1,#model.modules-1 do
        model.modules[i] = model.modules[i+1]
    end
    model.modules[#model.modules] = nil

    return createClassifierAndHasher(model, 2048)
end

function getTextModel2()
    local model = loadcaffe.load(filePath .. 'text model/tag_trainnet.prototxt', filePath .. 'text model/snapshot_iter_200.caffemodel', 'cudnn')

    -- Remove first layer that comes from caffemodel
    model.modules[1] = nil
    for i = 1,#model.modules-1 do
        model.modules[i] = model.modules[i+1]
    end
    model.modules[#model.modules] = nil

    model:add(nn.Sigmoid())
    return model
end

function getImageModel2()

    local model = loadcaffe.load(filePath .. 'CNN Model/trainnet.prototxt', filePath .. 'CNN Model/snapshot_iter_16000.caffemodel', 'cudnn')
    model:add(nn.Sigmoid())
    return model
end


function createClassifierAndHasher(model, prevLayerSize)

    -- Grab classification layer and remove it
    local classLayer = nn.Sequential()
    classLayer:add(model.modules[#model.modules])
    classLayer:add(nn.Sigmoid())
    model.modules[#model.modules] = nil

    local hashLayer = nn.Sequential()
    hashLayer:add(nn.Linear(prevLayerSize, hashLayerSize)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer))
    hashLayer:add(getSoftMaxLayer())

    local concat = nn.ConcatTable()
    concat:add(classLayer)
    concat:add(hashLayer)

    model:add(concat)

    local classifier = nn.Sequential()
    classifier:add(model)
    classifier:add(nn.SelectTable(1))

    local hasher = nn.Sequential()
    hasher:add(model)
    hasher:add(nn.SelectTable(2))

    return classifier, hasher
end

function createCombinedModel(imageHasher, textHasher)

    local model = nn.Sequential()

    cnn_text = nn.ParallelTable()
    cnn_text:add(imageHasher)
    cnn_text:add(textHasher)

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