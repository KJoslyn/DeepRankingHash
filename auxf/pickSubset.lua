-- function getKFoldSplit(kFold_pos_pairs, kFold_neg_pairs, pos_pair_splits, neg_pair_splits, kFold_images, kFold_texts, kNum)
function getKFoldSplit(kNum)

	local K = kFold_images:size(1)

	trainImages = torch.LongTensor()
	trainTexts = torch.LongTensor()
	valImages = torch.LongTensor()
	valTexts = torch.LongTensor()
	pos_pairs = torch.Tensor()
	neg_pairs = torch.LongTensor()

	for k = 1,K do
		if k == kNum then
			valImages = kFold_images[k]
			valTexts = kFold_texts[k]
		else
			trainImages = trainImages:cat(kFold_images[k])
			trainTexts = trainTexts:cat(kFold_texts[k])
			pos_pairs = pos_pairs:cat(kFold_pos_pairs[ {{ pos_pair_splits[k], pos_pair_splits[k+1] - 1 }} ], 1)
			neg_pairs = neg_pairs:cat(kFold_neg_pairs[ {{ neg_pair_splits[k], neg_pair_splits[k+1] - 1 }} ], 1)
		end
	end

	return trainImages, trainTexts, valImages, valTexts, pos_pairs, neg_pairs
end

function pickKFoldSubset(splitSize, K)

	local totSize = splitSize * K

	if not sim_ratio_tr then
		sim_ratio_tr = torch.load(filePath .. 'simRatioTr.t7')
	end

	totImages = torch.randperm(17251)
	totTexts = torch.randperm(17251)

	kFold_images = torch.LongTensor(K, splitSize)
	kFold_texts = torch.LongTensor(K, splitSize)

	local k = 0
	for k = 1, K do
		local startIdx = (k-1)*splitSize + 1
		local endIdx = (k)*splitSize
		kFold_images[k] = totImages[ {{ startIdx, endIdx }} ]
		kFold_texts[k] = totTexts[ {{ startIdx, endIdx }} ]
    end

	kFold_pos_pairs = torch.Tensor(splitSize * splitSize * K, 3)
	kFold_neg_pairs = torch.LongTensor(splitSize * splitSize * K, 2)

	pos_pair_splits = torch.LongTensor(K+1)
	neg_pair_splits = torch.LongTensor(K+1)
	pos_pair_splits[1] = 1
	neg_pair_splits[1] = 1

	p_idx = 1
	n_idx = 1
	for k = 1, K do
		for i = 1, splitSize do
			for j = 1, splitSize do
			    local sr = sim_ratio_tr[kFold_images[k][i]][kFold_texts[k][j]]
				if sr == 0 then
					kFold_neg_pairs[n_idx][1] = kFold_images[k][i]
					kFold_neg_pairs[n_idx][2] = kFold_texts[k][j]
					n_idx = n_idx + 1
				elseif sr > 0.5 then
					kFold_pos_pairs[p_idx][1] = kFold_images[k][i]
					kFold_pos_pairs[p_idx][2] = kFold_texts[k][j]
					pos_pairs[k][p_idx][3] = sr
					p_idx = p_idx + 1
				end
			end
        end
		pos_pair_splits[k+1] = p_idx
		neg_pair_splits[k+1] = n_idx
    end
	kFold_pos_pairs:resize(p_idx - 1, 3)
	kFold_neg_pairs:resize(n_idx - 1, 2)
end

function pickSubset(loadPairsFromFile)

	-- trainset[1] is all images
	-- trainset[2] is all texts
	if not loadPairsFromFile then

		sim_ratio_tr = torch.load(filePath .. 'simRatioTr.t7')
		sim_ratio_te = torch.load(filePath .. 'simRatioTe.t7')

		local images = torch.randperm(17251)[{{1,6000}}]:long()
		local texts = torch.randperm(17251)[{{1,6000}}]:long()

		trainImages = images[ {{ 1, 5000 }} ]
		trainTexts = texts[ {{ 1, 5000 }} ]

		valImages = images[ {{ 5001, 6000 }} ]
		valTexts = texts[ {{ 5001, 6000 }} ]

		pos_pairs = torch.Tensor(5000*5000, 3)
		neg_pairs = torch.LongTensor(5000*5000, 2)
		p_idx = 1
		n_idx = 1
		for i = 1,5000 do
			for j = 1,5000 do
				local sr = sim_ratio_tr[trainImages[i]][trainTexts[j]]
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

		subset_info = {}
		subset_info.pos_pairs = pos_pairs -- Important- this is loaded later as pos_pairs full
		subset_info.neg_pairs = neg_pairs -- Important- this is loaded later as neg_pairs full
		subset_info.trainImages = trainImages
		subset_info.trainTexts = trainTexts
		subset_info.valImages = valImages
		subset_info.valTexts = valTexts
	else
		subset_info = torch.load(filePath .. 'subsetInfo.t7')

		pos_pairs_full = subset_info.pos_pairs
		neg_pairs_full = subset_info.neg_pairs
		trainImages = subset_info.trainImages
		trainTexts = subset_info.trainTexts
		valImages = subset_info.valImages
		valTexts = subset_info.valTexts

		pos_pairs_image = subset_info.pos_pairs_image
		neg_pairs_image = subset_info.neg_pairs_image
		pos_pairs_text = subset_info.pos_pairs_text
		neg_pairs_text = subset_info.neg_pairs_text
	end

	local p_size = pos_pairs_full:size(1)
	local n_size = neg_pairs_full:size(1)

	return pos_pairs_full, neg_pairs_full, trainImages, trainTexts, valImages, valTexts, pos_pairs_image, neg_pairs_image, pos_pairs_text, neg_pairs_text
end

-- pos_pairs, neg_pairs, trainImages, trainTexts, valImages, valTexts, p_size, n_size = pickSubset(true)

-- -- This method does not allow replacement
-- tot_size = pos_pairs:size(1) + neg_pairs:size(1)
-- pair_perm = torch.randperm(tot_size)

-- pos_perm = torch.randperm(p_size)
-- neg_perm = torch.randperm(n_size)
-- epoch_pos_idx = pos_perm[{{1,posExamplesPerEpoch}}]
-- epoch_neg_idx = neg_perm[{{1,negExamplesPerEpoch}}]

-- y = torch.Tensor(100):fill(L)
-- y = y:cat(torch.Tensor(500):fill(0))

-- This takes forever, although replacement is possible
-- p_range = torch.range(1,p_size)
-- n_range = torch.range(1,n_size)
-- epoch_pos_idx = torch.multinomial(p_range, 100, false)
-- epoch_neg_idx = torch.multinomial(n_range, 500, false)

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

	sim_ratio_tr = torch.load(filePath .. 'simRatioTr.t7')

	subset_info = torch.load(filePath .. 'subsetInfo.t7')

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

	torch.save(filePath .. 'subsetInfo.t7', subset_info)

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