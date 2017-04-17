require 'image'

function dispRetrievedForQuery(retrievalFileName, queryIdx)

    file = io.open(retrievalFileName)
    target = string.format("Query%d", queryIdx)
    line = file:read("*line")
    line = string.gsub(line, "%s+", "")
    while line and line ~= target do
        line = file:read("*line")
        line = string.gsub(line, "%s+", "")
    end
    for i=1,10 do
        num = file:read("*line")
        num = string.gsub(num, "%s+", "")
        RR(tonumber(num))
    end
    io.close(file)
end

function EE(idxInTestset)
    if not testFileList then
        testTxtFile = io.open('/home/kejosl/kevin/mirflickr_test.txt')
        imageFile = testTxtFile:read("*line")
        testFileList = {}

        while imageFile do
            imageFile = string.gsub(imageFile, "%s+", "") 
            testFileList[#testFileList + 1] = imageFile
            imageFile = testTxtFile:read("*line")
        end
        io.close(testTxtFile)
    end

    im = image.load('/home/kejosl/Datasets/mirflickr/' .. testFileList[idxInTestset])
    te = image.toDisplayTensor(im)
    image.display(te)
end

function RR(idxInTrainset)
    if not trainFileList then
        trainTxtFile = io.open('/home/kejosl/kevin/mirflickr_train.txt')
        imageFile = trainTxtFile:read("*line")
        trainFileList = {}

        while imageFile do
            imageFile = string.gsub(imageFile, "%s+", "") 
            trainFileList[#trainFileList + 1] = imageFile
            imageFile = trainTxtFile:read("*line")
        end
        io.close(trainTxtFile)
    end

    im = image.load('/home/kejosl/Datasets/mirflickr/' .. trainFileList[idxInTrainset])
    te = image.toDisplayTensor(im)
    image.display(te)
end 

function displayTrainOrTestImage(idxInTrainOrTestset, trainOrTest)

    if trainOrTest == 'R' then -- train
        RR(idxInTrainOrTestset)
    elseif trainOrTest == 'E' then -- test
        EE(idxInTrainOrTestset)
    else
        print("Train: R, Test: E")
    end
end