local nninit = require 'nninit'

function getHashLayerFullyConnected(prevLayerSize, hashLayerSize, lrMultForHashLayer, addHiddenLayer)

    local model = nn.Sequential()

    if addHiddenLayer then
        model:add(nn.Linear(prevLayerSize, prevLayerSize)
                 :init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'})
                 :learningRate('weight', lrMultForHashLayer))
        model:add(cudnn.ReLU(true))
        model:add(nn.Dropout(0.500000))
    end

    model:add(nn.Linear(prevLayerSize, hashLayerSize)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer))

    model:add(nn.Reshape(L, k)) -- TODO: Change 120 to batchSize or inputSize
    model:add(nn.SplitTable(2))

    map = nn.MapTable()
    map:add(nn.SoftMax())

    model:add(map)

    model:add(nn.JoinTable(2))

    return model
end

function getHashLayerGrouped(prevLayerSize, L, k, lrMultForHashLayer, addHiddenLayer)

    local groupSize = prevLayerSize / L

    local hashLayer = nn.Sequential()

    if addHiddenLayer then
        hashLayer:add(nn.Linear(prevLayerSize, prevLayerSize)
                 :init('weight', nninit.xavier, {dist = 'normal'})
                 :learningRate('weight', lrMultForHashLayer))
    end

    hashLayer:add(nn.Reshape(L, groupSize))
    hashLayer:add(nn.SplitTable(2))

    map1 = nn.MapTable()
    map1:add(nn.Linear(groupSize, k)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer))

    map2 = nn.MapTable()
    map2:add(nn.SoftMax())

    hashLayer:add(map1)
    hashLayer:add(map2)

    hashLayer:add(nn.JoinTable(2))

    return hashLayer
end

function getImageModel()

    -- local model = loadcaffe.load(filePath .. 'CNN Model/trainnet.prototxt', filePath .. 'CNN Model/snapshot_iter_16000.caffemodel', 'cudnn')
    local model = nn.Sequential()

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

    return model
end

function getImageModelForFullNet(L, k, type, lrMultForHashLayer)

    local model = getImageModel()

    local snapshot2ndLevelDir = 'imageNet'
    local snapshotFile = 'snapshot_epoch_500.t7'
    loadModelSnapshot(model, snapshot2ndLevelDir, snapshotFile)

    model.modules[#model.modules] = nil -- This is messy, but need to remove sigmoid layer for now. Will add it back later.
    return createClassifierAndHasher(model, 4096, L, k, type, lrMultForHashLayer)
end

function getTextModelForFullNet(L, k, type, lrMultForHashLayer)

    local model = loadcaffe.load(filePath .. 'text model/tag_trainnet.prototxt', filePath .. 'text model/snapshot_iter_200.caffemodel', 'cudnn')

    -- Remove first layer that comes from caffemodel
    model.modules[1] = nil
    for i = 1,#model.modules-1 do
        model.modules[i] = model.modules[i+1]
    end
    model.modules[#model.modules] = nil

    return createClassifierAndHasher(model, 2048, L, k, type, lrMultForHashLayer)
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


function createClassifierAndHasher(model, prevLayerSize, L, k, type, lrMultForHashLayer)

    -- Grab classification layer and remove it
    local classLayer = nn.Sequential()
    classLayer:add(model.modules[#model.modules])
    classLayer:add(nn.Sigmoid())
    model.modules[#model.modules] = nil

    local hashLayer
    if type == 'hfc' then
        hashLayer = getHashLayerFullyConnected(prevLayerSize, L*k, lrMultForHashLayer, true) 
    elseif type == 'fc' then
        hashLayer = getHashLayerFullyConnected(prevLayerSize, L*k, lrMultForHashLayer, false) 
    elseif type == 'hgr' then
        hashLayer = getHashLayerGrouped(prevLayerSize, L, k, lrMultForHashLayer, true)
    elseif type == 'gr' then
        hashLayer = getHashLayerGrouped(prevLayerSize, L, k, lrMultForHashLayer, false)
    else
        print('ERROR: Unrecognized hash layer type')
    end

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

function getSiameseHasher(hasher)

    local model = nn.Sequential()

    -- local net2 = hasher:clone('weight', 'bias', 'gradWeight', 'gradBias') -- Don't do this and call share later
    local net2 = hasher:clone()

    local prl = nn.ParallelTable()
    prl:add(hasher)
    prl:add(net2)

    model:add(prl)
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

function loadModelSnapshot(model, snapshot2ndLevelDir, snapshotFileName)


  -- If these aren't specified, use hardcoded values
  if not snapshot2ndLevelDir and snapshotFileName then
    -- local snapshot2ndLevelDir = 'Lr5e4_5kquery_5kdatabase'
    local snapshot2ndLevelDir = 'imageNet'
    local snapshotFileName = 'snapshot_epoch_500.t7'
  end

  print('****Loading snapshot: ' .. snapshot2ndLevelDir .. '/' .. snapshotFileName)

  snapshotFullPath = snapshotDir .. '/' .. snapshot2ndLevelDir .. '/' .. snapshotFileName
  snapshot = torch.load(snapshotFullPath)
  N = snapshot.params:size(1)

  local params, gparams = model:getParameters()

  batchSize = 1e5

  local numBatches = torch.ceil(N / batchSize)
  for b = 0, numBatches - 1 do
      startIndex = b * batchSize + 1
      endIndex = math.min((b + 1) * batchSize, N)
      params[ {{ startIndex, endIndex }} ]:copy(snapshot.params[ {{ startIndex, endIndex }} ])
  end

end

