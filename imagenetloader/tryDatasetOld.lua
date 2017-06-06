require 'dataset'
local ffi = require 'ffi'

-- dataset = imageLoader{path='/home/kjoslyn/datasets/mirflickr', split=100, sampleSize={3,227,227}, splitFolders={'training'}}
dataset = imageLoader{path='/home/kejosl/Datasets/mirflickr/ImageData', split=100, sampleSize={3,227,227}, splitFolders={'training'}}

-- print('Saving to test.t7')
-- torch.save('test.t7', dataset)

print('Class names', dataset.classes)
print('Total images in dataset: ' .. dataset:size())
print('Training size: ' .. dataset:sizeTrain())
print('Testing size: ' .. dataset:sizeTest())
print()
for k,v in ipairs(dataset.classes) do
   print('Images for class: ' .. v .. ' : ' .. dataset:size(k) .. ' == ' .. dataset:size(v))
   print('Training size: ' .. dataset:sizeTrain(k))
   print('Testing size: ' .. dataset:sizeTest(k))
   print('First image path: ' .. ffi.string(dataset.imagePath[dataset.classList[k][1]]:data()))
   print('Last image path: ' .. ffi.string(dataset.imagePath[dataset.classList[k][#dataset.classList[k]]]:data()))
   print()
end

-- now sample from this dataset and print out sizes, also visualize
print('Getting 128 training samples')
local inputs, labels = dataset:sample(128)
print('Size of 128 training samples')
print(#inputs)
print('Size of 128 training labels')
print(#labels)


print('Getting 1 training sample')
local inputs, labels = dataset:sample()
print('Size of 1 training sample')
print(#inputs)
print('1 training label: ' .. labels)

print('Getting 2 training samples')
local inputs, labels = dataset:sample(2)
print('Size of 2 training samples')
print(#inputs)
print('Size of 1 training labels')
print(#labels)

-- print('Getting test samples')
-- local count = 0
-- for inputs, labels in dataset:test(128) do
--    print(#inputs)
--    print(#labels)
--    count = count + 1
--    print(count)
-- end





-- dataset = chex.dataset{paths={'/home/fatbox/data/imagenet12/cropped_quality100/train'}, sampleSize={3,200,200}}
-- dataset = chex.dataset{paths={'/home/fatbox/data/imagenet-fall11/images'}, sampleSize={3,200,200}}

-- dataset = chex.dataset({'asdsd'}, {3,200,200})
