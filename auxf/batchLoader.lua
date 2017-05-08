function getBatch(pos_pairs, neg_pairs, modality)

    local p_size = pos_pairs:size(1)
    local n_size = neg_pairs:size(1)

    -- Choose positive and negative examples
    idxInPosPairs = torch.rand(p.posExamplesPerBatch):mul(p_size):ceil():long()
    idxInNegPairs = torch.rand(p.negExamplesPerBatch):mul(n_size):ceil():long()

    batchPosPairs = pos_pairs:index(1, idxInPosPairs)
    batchNegPairs = neg_pairs:index(1, idxInNegPairs)

    batchIdxInTrainset = torch.cat(batchPosPairs:sub(1,p.posExamplesPerBatch,1,2):long(), batchNegPairs, 1)

    batch_sim_label_for_loss = torch.Tensor(p.batchSize):fill(0)
    for pos = 1, p.posExamplesPerBatch do
        batch_sim_label_for_loss[pos] = batchPosPairs[pos][3]
    end
    batch_sim_label_for_loss = batch_sim_label_for_loss * p.L

    batch = {}
    batch.data = {}
    batch.label = {}

    if modality == 'C' then -- Cross Modal (Both modalities)
        batch.data[I] = d.trainset[I]:index(1, batchIdxInTrainset:select(2,1)) -- TODO: Fix long conversion in root pos_pairs and neg_pairs
        batch.data[X] = d.trainset[X]:index(1, batchIdxInTrainset:select(2,2))
        batch.label[I] = d.train_labels_image:index(1, batchIdxInTrainset:select(2,1))
        batch.label[X] = d.train_labels_text:index(1, batchIdxInTrainset:select(2,2))
    elseif modality == 'I' then -- Image intramodal
        batch.data[1] = d.trainset[I]:index(1, batchIdxInTrainset:select(2,1))
        batch.data[2] = d.trainset[I]:index(1, batchIdxInTrainset:select(2,2))
        batch.label[1] = d.train_labels_image:index(1, batchIdxInTrainset:select(2,1))
        batch.label[2] = d.train_labels_image:index(1, batchIdxInTrainset:select(2,2))
    elseif modality == 'X' then -- Text intramodal
        batch.data[1] = d.trainset[X]:index(1, batchIdxInTrainset:select(2,1))
        batch.data[2] = d.trainset[X]:index(1, batchIdxInTrainset:select(2,2))
        batch.label[1] = d.train_labels_text:index(1, batchIdxInTrainset:select(2,1))
        batch.label[2] = d.train_labels_text:index(1, batchIdxInTrainset:select(2,2))
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