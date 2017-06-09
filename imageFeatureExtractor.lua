require 'fullNet'

function getImageFeaturesForLinearMethods(datasetType)

    loadParamsAndPackages(datasetType, 100) -- second param is iterations per epoch, not needed here

    local imDir = '/home/kjoslyn/kevin/Project/IMAGENET/'
    
    m.featureExtractor = loadcaffe.load(imDir .. 'deploy.prototxt', imDir .. 'bvlc_alexnet.caffemodel', 'cudnn')

    for i = 21,24 do
        m.featureExtractor.modules[i] = nil
    end

    local imageRootPath = g.datasetPath .. 'ImageData'

    d.dataset = imageLoader{path=imageRootPath, sampleSize={3,227,227}, splitFolders={'training', 'pretraining', 'val', 'query'}}

    local Ndb = d.dataset:sizeTrain() + d.dataset:sizePretraining() + d.dataset:sizeVal()
    local Ntr = d.dataset:sizeTrain() + d.dataset:sizeVal()
    local Nte = d.dataset:sizeTest()

    m.featureExtractor:evaluate()

    local trFeats, trTags, trLabels = calcFeatures({'training'}, d.dataset:sizeTrain())
    local valFeats, valTags, valLabels = calcFeatures({'val'}, d.dataset:sizeVal())
    local preFeats, preTags, preLabels = calcFeatures({'pretraining'}, d.dataset:sizePretraining())
    local teFeats, teTags, teLabels = calcFeatures({'query'}, d.dataset:sizeTest())

    trFeats = torch.cat(trFeats, valFeats, 1)
    trTags = torch.cat(trTags, valTags, 1)
    trLabels = torch.cat(trLabels, valLabels, 1)
    local dbFeats = torch.cat(trFeats, preFeats, 1)
    local dbTags = torch.cat(trTags, preTags, 1)
    local dbLabels = torch.cat(trLabels, preLabels, 1)

    -- local dbFeats = torch.cat(trFeats, valFeats, 1)
    -- local dbTags = torch.cat(trTags, valTags, 1)
    -- local dbLabels = torch.cat(trLabels, valLabels, 1)
    -- local dbFeats = torch.cat(dbFeats, preFeats, 1)
    -- local dbTags = torch.cat(dbTags, preTags, 1)
    -- local dbLabels = torch.cat(dbLabels, preLabels, 1)

    -- local dbClasses = {'training', 'pretraining', 'val'}
    -- local trClasses = {'training', 'val'}
    -- local teClasses = {'query'}

    -- local dbFeats, dbLabels = calcFeatures(dbClasses, Ndb) -- TODO: Redundant calculation of training features
    -- local trFeats, trLabels = calcFeatures(trClasses, Ntr)
    -- local teFeats, teLabels = calcFeatures(teClasses, Nte)

    -- local dbTags = d.dataset:getBySplit(dbClasses, 'X', 1, Ndb)
    -- local trTags = d.dataset:getBySplit(trClasses, 'X', 1, Ntr)
    -- local teTags = d.dataset:getBySplit(teClasses, 'X', 1, Nte)

    local data = {}
    data.Ndb = Ndb
    data.Ntraining = Ntr
    data.Ntest = Nte
    data.Xdb = dbTags
    data.Xtrain = trTags
    data.Xtest = teTags
    data.Ydb = dbFeats
    data.Ytrain = trFeats
    data.Ytest = teFeats
    data.L_db = dbLabels
    data.L_tr = trLabels
    data.L_te = teLabels

    -- matio.save(g.datasetPath .. 'alexNetFeatures.mat', {data=data})

    return data
end

function calcFeatures(classes, N)

    local feats = torch.Tensor(N, 4096)
    local tags = torch.Tensor(N, p.tagDim)
    local labels = torch.FloatTensor(N, p.numClasses)

    local batchSize = 128
    local numBatches = math.ceil(N / batchSize)

    for batchNum = 0, numBatches - 1 do

        local startIndex = batchNum * batchSize + 1
        local endIndex = math.min((batchNum + 1) * batchSize, N)
        local thisBatchSize = endIndex - startIndex + 1

        batchIm, batchTags, batchLabels = d.dataset:getBySplit(classes, 'B', startIndex, endIndex)
        local batchFeats = m.featureExtractor:forward(batchIm:cuda())
        feats[{ {startIndex, endIndex} }] = torch.Tensor(thisBatchSize, 4096):copy(batchFeats)
        tags[{ {startIndex, endIndex} }] = batchTags
        labels[{ {startIndex, endIndex} }] = batchLabels
    end

    return feats, tags, labels
end