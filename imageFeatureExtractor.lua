require 'fullNet'

function getImageFeaturesForLinearMethods(datasetType, useage)

    local LINEAR_METHODS = 1
    local DEEP = 2

    loadParamsAndPackages(datasetType, 100) -- second param is iterations per epoch, not needed here

    if useage == LINEAR_METHODS then

        local imDir = '/home/kjoslyn/kevin/Project/IMAGENET/'
        
        m.featureExtractor = loadcaffe.load(imDir .. 'deploy.prototxt', imDir .. 'bvlc_alexnet.caffemodel', 'cudnn')

        for i = 21,24 do
            m.featureExtractor.modules[i] = nil
        end

    elseif useage == DEEP then

        local model = getFineTunedImageModel()

        for i = 21,24 do
            model.modules[i] = nil
        end
        m.featureExtractor = model:cuda()
    end

    local imageRootPath = g.datasetPath .. 'ImageData'

    d.dataset = imageLoader{path=imageRootPath, sampleSize={3,227,227}, splitFolders={'training', 'pretraining', 'val', 'query'}}

    local Ndb = d.dataset:sizeTrain() + d.dataset:sizePretraining() + d.dataset:sizeVal()
    local Ntr = d.dataset:sizeTrain() + d.dataset:sizeVal()
    local Nte = d.dataset:sizeTest()

    m.featureExtractor:evaluate()

    print('Calculating training features')
    local trFeats, trTags, trLabels = calcFeatures({'training'}, d.dataset:sizeTrain())
    print('Calculating val features')
    local valFeats, valTags, valLabels = calcFeatures({'val'}, d.dataset:sizeVal())
    print('Calculating pretraining features')
    local preFeats, preTags, preLabels = calcFeatures({'pretraining'}, d.dataset:sizePretraining())
    print('Calculating query features')
    local teFeats, teTags, teLabels = calcFeatures({'query'}, d.dataset:sizeTest())

    local data = {}
    if useage == LINEAR_METHODS then

        trFeats = torch.cat(trFeats, valFeats, 1)
        trTags = torch.cat(trTags, valTags, 1)
        trLabels = torch.cat(trLabels, valLabels, 1)
        local dbFeats = torch.cat(trFeats, preFeats, 1)
        local dbTags = torch.cat(trTags, preTags, 1)
        local dbLabels = torch.cat(trLabels, preLabels, 1)

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

        matio.save(g.datasetPath .. 'alexNetFeatures_linear.mat', {data=data})

    elseif useage == DEEP then

        data.trainset = {}
        data.trainset[I] = {}
        data.trainset[X] = {}
        data.valset = {}
        data.valset[I] = {}
        data.valset[X] = {}
        data.testset = {}
        data.testset[I] = {}
        data.testset[X] = {}
        data.pretrainset = {}
        data.pretrainset[I] = {}
        data.pretrainset[X] = {}

        data.trainset[I].data = trFeats
        data.trainset[I].label = trLabels
        data.trainset[X].data = trTags
        data.trainset[X].label = trLabels

        data.valset[I].data = valFeats
        data.valset[I].label = valLabels
        data.valset[X].data = valTags
        data.valset[X].label = valLabels

        data.pretrainset[I].data = preFeats
        data.pretrainset[I].label = preLabels
        data.pretrainset[X].data = preTags
        data.pretrainset[X].label = preLabels

        data.testset[I].data = teFeats
        data.testset[I].label = teLabels
        data.testset[X].data = teTags
        data.testset[X].label = teLabels

        torch.save(g.datasetPath .. 'alexNetFeatures_deep.t7', data)
    end

    return data
end

function calcFeatures(classes, N)

    local feats = torch.Tensor(N, 4096)
    local tags = torch.Tensor(N, p.tagDim)
    local labels = torch.FloatTensor(N, p.numClasses)

    local batchSize = 1000
    local numBatches = math.ceil(N / batchSize)

    for batchNum = 0, numBatches - 1 do

        if batchNum % 5 == 0 then
            print(string.format('Batch %d', batchNum))
        end

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