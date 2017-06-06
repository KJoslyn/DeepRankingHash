-- TODO: This is the only function currently used in pickSubset. Technically this could just be a .m script.
function getCrossModalPairs()

	if not matio then
		matio = require 'matio'
	end
	-- srt = sim_ratio_tr
	-- local srt = torch.load(g.datasetPath .. 'simRatioTr.t7')
	local srt = matio.load(g.datasetPath .. 'simRatioTr.mat')
	srt = srt.sim_ratio
	local N = srt:size(1)

	-- local split = matio.load(g.datasetPath .. 'datasetSplit.mat')
	-- split = split.split

	local pos_pairs_full = torch.Tensor(N*N, 3)
	local neg_pairs_full = torch.LongTensor(N*N, 2)
	local p_idx = 1
	local n_idx = 1
	for i = 1,N do
		for j = 1,N do
			local sr = srt[i][j]
			-- local iIdx = split.trainSet.indices[1][i]
			-- local jIdx = split.trainSet.indices[1][j]
			if sr == 0 then
				neg_pairs_full[n_idx][1] = i
				neg_pairs_full[n_idx][2] = j
				n_idx = n_idx + 1
			elseif sr > 0.5 then
				pos_pairs_full[p_idx][1] = i
				pos_pairs_full[p_idx][2] = j
				pos_pairs_full[p_idx][3] = sr
				p_idx = p_idx + 1
			end
		end
		if i % 1000 == 0 then
		   print('Outer loop: done with ' .. i)
        end
	end
	pos_pairs_full:resize(p_idx - 1, 3)
	neg_pairs_full:resize(n_idx - 1, 2)

	subset_info = {}
	subset_info.pos_pairs = pos_pairs_full -- Important- this is loaded later as pos_pairs full
	subset_info.neg_pairs = neg_pairs_full -- Important- this is loaded later as neg_pairs full

	local p_size = pos_pairs_full:size(1)
	local n_size = neg_pairs_full:size(1)

	return pos_pairs_full, neg_pairs_full
end

-- function getKFoldSplit(kFold_images, kFold_texts, kNum)
function getKFoldSplit(kNum)

	-- Currently, k fold split is not compatable with siamese models.

	local K = d.kFold_images:size(1)
	local splitSize = d.kFold_images:size(2)

	local trainImages = torch.LongTensor()
	local trainTexts = torch.LongTensor()
	local valImages = torch.LongTensor()
	local valTexts = torch.LongTensor()

	for k = 1,K do
		if k == kNum then
			valImages = d.kFold_images[k]
			valTexts = d.kFold_texts[k]
		else
			trainImages = trainImages:cat(d.kFold_images[k])
			trainTexts = trainTexts:cat(d.kFold_texts[k])
		end
	end

	if not d.sim_ratio_tr then
		-- In both cases, only examples with > 0.5 similarity will be added to pos_pairs.
		-- However, only the non-byte version works with variable sim_label_type
		if p.sim_label_type == 'fixed' then
			d.sim_ratio_tr = torch.load(g.filePath .. 'simRatioTrByte.t7')
        else
			d.sim_ratio_tr = torch.load(g.filePath .. 'simRatioTr.t7')
		end
	end

	local Ntrain = trainImages:size(1)
	local pos_pairs
	if p.sim_label_type == 'fixed' then
		pos_pairs = torch.LongTensor(math.pow(Ntrain, 2), 3)
    else
		pos_pairs = torch.Tensor(math.pow(Ntrain, 2), 3)
	end
	local neg_pairs = torch.LongTensor(math.pow(Ntrain, 2), 2)
	local p_idx = 1
	local n_idx = 1
	for i = 1, Ntrain do
		for j = 1, Ntrain do
			local sr = d.sim_ratio_tr[trainImages[i]][trainTexts[j]]
			if sr == 0 then
				neg_pairs[n_idx][1] = trainImages[i]
				neg_pairs[n_idx][2] = trainTexts[j]
				n_idx = n_idx + 1
			elseif sr > 0.5 then
				pos_pairs[p_idx][1] = trainImages[i]
				pos_pairs[p_idx][2] = trainTexts[j]
				pos_pairs[p_idx][3] = sr
				p_idx = p_idx + 1
			end
		end
	end

	pos_pairs:resize(p_idx - 1, 3)
	neg_pairs:resize(n_idx - 1, 2)

    return pos_pairs, neg_pairs, trainImages, trainTexts, valImages, valTexts
end

function pickKFoldSubset(splitSize, K, loadPairsFromFile)

	if loadPairsFromFile then
		local subset_info = torch.load(g.filePath .. 'kFoldSubsetInfo.t7')
		d.kFold_images = subset_info.kFold_images
		d.kFold_texts = subset_info.kFold_texts
	else
		local totImages = torch.randperm(17251)
		local totTexts = torch.randperm(17251)

		d.kFold_images = torch.LongTensor(K, splitSize)
		d.kFold_texts = torch.LongTensor(K, splitSize)

		local k = 0
		for k = 1, K do
			local startIdx = (k-1)*splitSize + 1
			local endIdx = (k)*splitSize
			d.kFold_images[k] = totImages[ {{ startIdx, endIdx }} ]
			d.kFold_texts[k] = totTexts[ {{ startIdx, endIdx }} ]
		end
    end

	-- return kFold_images, kFold_texts
end

function pickSubset(loadPairsFromFile)

	-- d.trainset[1] is all images
	-- d.trainset[2] is all texts
	if not loadPairsFromFile then

		local sim_ratio_tr = torch.load(g.datasetPath .. 'simRatioTr.t7')
		-- sim_ratio_te = torch.load(g.filePath .. 'simRatioTe.t7')

		local images = torch.randperm(17251)[{{1,6000}}]:long()
		local texts = torch.randperm(17251)[{{1,6000}}]:long()

		local trainImages = images[ {{ 1, 5000 }} ]
		local trainTexts = texts[ {{ 1, 5000 }} ]

		local valImages = images[ {{ 5001, 6000 }} ]
		local valTexts = texts[ {{ 5001, 6000 }} ]

		local pos_pairs_full = torch.Tensor(5000*5000, 3)
		local neg_pairs_full = torch.LongTensor(5000*5000, 2)
		local p_idx = 1
		local n_idx = 1
		for i = 1,5000 do
			for j = 1,5000 do
				local sr = sim_ratio_tr[trainImages[i]][trainTexts[j]]
				if sr == 0 then
					neg_pairs_full[n_idx][1] = trainImages[i]
					neg_pairs_full[n_idx][2] = trainTexts[j]
					n_idx = n_idx + 1
				elseif sr > 0.5 then
					pos_pairs_full[p_idx][1] = trainImages[i]
					pos_pairs_full[p_idx][2] = trainTexts[j]
					pos_pairs_full[p_idx][3] = sr
					p_idx = p_idx + 1
				end
			end
		end

		pos_pairs_full:resize(p_idx - 1, 3)
		neg_pairs_full:resize(n_idx - 1, 2)

		subset_info = {}
		subset_info.pos_pairs = pos_pairs_full -- Important- this is loaded later as pos_pairs full
		subset_info.neg_pairs = neg_pairs_full -- Important- this is loaded later as neg_pairs full
		subset_info.trainImages = trainImages
		subset_info.trainTexts = trainTexts
		subset_info.valImages = valImages
		subset_info.valTexts = valTexts
	else
		subset_info = torch.load(g.datasetPath .. 'subsetInfo.t7')

		local pos_pairs_full = subset_info.pos_pairs
		local neg_pairs_full = subset_info.neg_pairs
		local trainImages = subset_info.trainImages
		local trainTexts = subset_info.trainTexts
		local valImages = subset_info.valImages
		local valTexts = subset_info.valTexts

		local pos_pairs_image = subset_info.pos_pairs_image
		local neg_pairs_image = subset_info.neg_pairs_image
		local pos_pairs_text = subset_info.pos_pairs_text
		local neg_pairs_text = subset_info.neg_pairs_text
	end

	local p_size = pos_pairs_full:size(1)
	local n_size = neg_pairs_full:size(1)

	return pos_pairs_full, neg_pairs_full, trainImages, trainTexts, valImages, valTexts, pos_pairs_image, neg_pairs_image, pos_pairs_text, neg_pairs_text
end

function pickValSet()

	valImages = torch.Tensor(1000)
	valTexts = torch.Tensor(1000)

	imagePerm = torch.randperm(6000) -- absolute worst case: first 5000 picked are in the trainset already, so need 1000 more
	local permIdx = 1
	local valIdx = 1
	while (valIdx <= 1000) do
		local im = imagePerm[permIdx]
		if torch.eq(trainImages, im):sum() == 0 then
			valImages[valIdx] = im
			valIdx = valIdx + 1
		end
		permIdx = permIdx + 1
	end

	textPerm = torch.randperm(6000)
	local permIdx = 1
	local valIdx = 1
	while (valIdx <= 1000) do
		local te = textPerm[permIdx]
		if torch.eq(trainTexts, te):sum() == 0 then
			valTexts[valIdx] = te
			valIdx = valIdx + 1
		end
		permIdx = permIdx + 1
	end
end

function addSimRatio()

	local sim_ratio_tr = torch.load(g.filePath .. 'simRatioTr.t7')

	local subset_info = torch.load(g.filePath .. 'subsetInfo.t7')

	local N = subset_info.pos_pairs:size(1)

	-- subset_info.pos_pairs = subset_info.pos_pairs:resize(N, 3)
	local pp = torch.Tensor(N, 3)
	for i = 1,N do
		pp[i][1] = subset_info.pos_pairs[i][1]
		pp[i][2] = subset_info.pos_pairs[i][2]
    end
	subset_info.pos_pairs = pp

	ii = 1
	while ii <= N do
		local sr = sim_ratio_tr[subset_info.pos_pairs[ii][1]][subset_info.pos_pairs[ii][2]]
		subset_info.pos_pairs[ii][3] = sr
		ii = ii + 1
	end

	torch.save(g.filePath .. 'subsetInfo.t7', subset_info)

end

function getIntraModalPosAndNegPairs(sim_ratio_tr, trainIorT)

	-- trainIorT should either be trainImages or trainTexts

	local s_pos_pairs = torch.Tensor(5000*5000, 3) -- s for "single" modality
	local s_neg_pairs = torch.LongTensor(5000*5000, 2)

	local p_idx = 1
	local n_idx = 1
	for i = 1,5000 do
		for j = 1,5000 do
			if i ~= j then
				local sr = sim_ratio_tr[trainIorT[i]][trainIorT[j]]
				if sr == 0 then
					s_neg_pairs[n_idx][1] = trainIorT[i]
					s_neg_pairs[n_idx][2] = trainIorT[j]
					n_idx = n_idx + 1
				elseif sr > 0.5 then
					s_pos_pairs[p_idx][1] = trainIorT[i]
					s_pos_pairs[p_idx][2] = trainIorT[j]
					s_pos_pairs[p_idx][3] = sr
					p_idx = p_idx + 1
				end
            end
		end
	end

	s_pos_pairs:resize(p_idx - 1, 3)
	s_neg_pairs:resize(n_idx - 1, 2)

	return s_pos_pairs, s_neg_pairs
end