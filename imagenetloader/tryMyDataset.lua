require 'dataset'
local ffi = require 'ffi'

function initDataset(splitFolders)
    dataset = imageLoader{path='/home/kejosl/datasets/mirflickr/ImageData/DCMH', sampleSize={3,224,224}, splitFolders=splitFolders}
end

initDataset({'training', 'query', 'val', 'pretraining'})
-- da, la = dataset:getBySplit({'training', 'query'},1,128);

-- dataset = imageLoader{path='/home/kjoslyn/datasets/mirflickr', sampleSize={3,227,227}, splitFolders={'training'}}