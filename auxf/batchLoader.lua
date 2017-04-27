function getBatch(pos_pairs, neg_pairs, modality)

    local p_size = pos_pairs:size(1)
    local n_size = neg_pairs:size(1)

    -- Choose positive and negative examples
    idxInPosPairs = torch.rand(posExamplesPerBatch):mul(p_size):ceil():long()
    idxInNegPairs = torch.rand(negExamplesPerBatch):mul(n_size):ceil():long()

    batchPosPairs = pos_pairs:index(1, idxInPosPairs)
    batchNegPairs = neg_pairs:index(1, idxInNegPairs)

    batchIdxInTrainset = torch.cat(batchPosPairs:sub(1,posExamplesPerBatch,1,2):long(), batchNegPairs, 1)

    batch_sim_label_for_loss = torch.Tensor(totNumExamplesPerBatch):fill(0)
    for p = 1, posExamplesPerBatch do
        batch_sim_label_for_loss[p] = batchPosPairs[p][3]
    end
    batch_sim_label_for_loss = batch_sim_label_for_loss * L

    batch = {}
    batch.data = {}
    batch.label = {}

    if modality == 'C' then -- Cross Modal (Both modalities)
        batch.data[I] = trainset[I]:index(1, batchIdxInTrainset:select(2,1)) -- TODO: Fix long conversion in root pos_pairs and neg_pairs
        batch.data[X] = trainset[X]:index(1, batchIdxInTrainset:select(2,2))
        batch.label[I] = train_labels_image:index(1, batchIdxInTrainset:select(2,1))
        batch.label[X] = train_labels_text:index(1, batchIdxInTrainset:select(2,2))
    elseif modality == 'I' then -- Image intramodal
        batch.data[1] = trainset[I]:index(1, batchIdxInTrainset:select(2,1))
        batch.data[2] = trainset[I]:index(1, batchIdxInTrainset:select(2,2))
        batch.label[1] = train_labels_image:index(1, batchIdxInTrainset:select(2,1))
        batch.label[2] = train_labels_image:index(1, batchIdxInTrainset:select(2,2))
    elseif modality == 'X' then -- Text intramodal
        batch.data[1] = trainset[X]:index(1, batchIdxInTrainset:select(2,1))
        batch.data[2] = trainset[X]:index(1, batchIdxInTrainset:select(2,2))
        batch.label[1] = train_labels_text:index(1, batchIdxInTrainset:select(2,1))
        batch.label[2] = train_labels_text:index(1, batchIdxInTrainset:select(2,2))
    else
        print("Error: unrecognized modality in getBatch")
    end

    batch.data[1] = batch.data[1]:cuda()
    batch.data[2] = batch.data[2]:cuda()
    batch.label[1] = batch.label[1]:cuda()
    batch.label[2] = batch.label[2]:cuda()
    batch.batch_sim_label_for_loss = batch_sim_label_for_loss:cuda()

    return batch
end

--[[
function getEpochPerm(epoch)

    epoch_pos_perm = pos_perm[ {{ epoch*posExamplesPerEpoch + 1 , (epoch + 1)*posExamplesPerEpoch }} ]
    epoch_neg_perm = neg_perm[ {{ epoch*negExamplesPerEpoch + 1 , (epoch + 1)*negExamplesPerEpoch }} ]

    return epoch_pos_perm, epoch_neg_perm
end

function getBatchPosOrNeg(array, batchNum, perm, batchSize)

    startIndex = batchNum * batchSize + 1
    endIndex = (batchNum + 1) * batchSize

    return array:index(1, perm[ {{ startIndex, endIndex }} ])
end

function getBatch(batchNum, epoch_pos_perm, epoch_neg_perm)

    pos_batch = getBatchPosOrNeg(pos_pairs, batchNum, epoch_pos_perm, posExamplesPerBatch)
    neg_batch = getBatchPosOrNeg(neg_pairs, batchNum, epoch_neg_perm, negExamplesPerBatch)

    batch_idx = torch.cat(pos_batch, neg_batch, 1)
    -- batch_idx = neg_batch -- TODO: Remove (for debugging)

    batch = {}
    batch.data = {}
    batch.data[I] = trainset[I]:index(1, batch_idx:select(2,1):long()) -- TODO: Fix long conversion in root pos_pairs and neg_pairs
    batch.data[X] = trainset[X]:index(1, batch_idx:select(2,2):long())
    batch.data[I] = batch.data[I]:cuda()
    batch.data[X] = batch.data[X]:cuda()

    setmetatable(batch, 
        {__index = function(t, i) 
                        return {t.data[I][i], t.data[X][i]} 
                    end}
    );

    function batch:size() 
        return self.data[1]:size(1) 
    end

    return batch, batch_idx
end
--]]