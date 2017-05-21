require 'dataset'
local ffi = require 'ffi'

function initDataset(splitFolders)
    dataset = imageLoader{path='/home/kejosl/Datasets/mirflickr/ImageData', split=100, sampleSize={3,227,227}, splitFolders=splitFolders}
end

-- dataset = imageLoader{path='/home/kjoslyn/datasets/mirflickr', split=100, sampleSize={3,227,227}, splitFolders={'training'}}