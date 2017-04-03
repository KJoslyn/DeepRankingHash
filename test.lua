require 'nn'
require 'loadcaffe' -- doesn't work on server
require 'image'
require 'optim'
require 'nnlr'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'auxFunctions'
require 'createModel'

filePath = '/home/kjoslyn/kevin/' -- server

I = 1
X = 2

trainset = {}
testset = {}

tm = getTextModel2()
im = getImageModel2()

getImageData()
getTextData()

-- trainset[X], trainset[I], testset[X], testset[I]
-- train_labels_image, test_labels_text

im = im:cuda()
tm = tm:cuda()

tm:evaluate()
im:evaluate()

testset[I] = testset[I]:cuda()
testset[X] = testset[X]:cuda()
test_labels_image = test_labels_image:cuda()
test_labels_text = test_labels_text:cuda()

imageAccuracy = calcClassAccuracy(im, testset[I], test_labels_image:cuda())
textAccuracy = calcClassAccuracy(tm, testset[X], test_labels_text:cuda())

print(string.format('Testset Image accuracy: %.2f', imageAccuracy))
print(string.format('Testset Text accuracy: %.2f', textAccuracy))

for b = 0,19 do
    local startIdx = b*120 + 1
    local endIdx = b*120 + 120

    trainBatch = trainset[X][ {{ startIdx, endIdx }} ]:cuda()
    trainBatch_labels = train_labels_text[ {{ startIdx, endIdx }} ]:cuda()

    textAccuracy = calcClassAccuracy(tm, trainBatch, trainBatch_labels)

    print(string.format('Batch Text accuracy: %.2f', textAccuracy))
end