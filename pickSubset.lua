function pickSubset(loadPairsFromFile)

	-- trainset[1] is all images
	-- trainset[2] is all texts
	if not loadPairsFromFile then

		sim_ratio_tr = torch.load(filePath .. 'simRatioTr.t7')
		sim_ratio_te = torch.load(filePath .. 'simRatioTe.t7')

		imageIdx = torch.randperm(17251)[{{1,5000}}]
		textIdx = torch.randperm(17251)[{{1,5000}}]

		pos_pairs = torch.Tensor(5000*5000, 2)
		neg_pairs = torch.Tensor(5000*5000, 2)
		p_idx = 1
		n_idx = 1
		for i = 1,5000 do
			for j = 1,5000 do
				if sim_ratio_tr[imageIdx[i]][textIdx[j]] == 0 then
					neg_pairs[n_idx][1] = imageIdx[i]
					neg_pairs[n_idx][2] = textIdx[j]
					n_idx = n_idx + 1
				elseif sim_ratio_tr[imageIdx[i]][textIdx[j]] > 0.5 then
					pos_pairs[p_idx][1] = imageIdx[i]
					pos_pairs[p_idx][2] = textIdx[j]
					p_idx = p_idx + 1
				end
			end
		end

		pos_pairs:resize(p_idx - 1, 2)
		neg_pairs:resize(n_idx - 1, 2)
	else
		train_pairs = torch.load(filePath .. 'trainPairs.t7')

		pos_pairs = train_pairs.pos_pairs
		neg_pairs = train_pairs.neg_pairs
	end

	p_size = pos_pairs:size(1)
	n_size = neg_pairs:size(1)

	return pos_pairs, neg_pairs, p_size, n_size
end

pos_pairs, neg_pars, p_size, n_size = pickSubset(true)

-- This method does not allow replacement
tot_size = pos_pairs:size(1) + neg_pairs:size(1)
pair_perm = torch.randperm(tot_size)

pos_perm = torch.randperm(p_size)
neg_perm = torch.randperm(n_size)
epoch_pos_idx = pos_perm[{{1,posExamplesPerEpoch}}]
epoch_neg_idx = neg_perm[{{1,negExamplesPerEpoch}}]

-- y = torch.Tensor(100):fill(L)
-- y = y:cat(torch.Tensor(500):fill(0))

-- This takes forever, although replacement is possible
-- p_range = torch.range(1,p_size)
-- n_range = torch.range(1,n_size)
-- epoch_pos_idx = torch.multinomial(p_range, 100, false)
-- epoch_neg_idx = torch.multinomial(n_range, 500, false)