function [ output_args ] = plotResults( evalInterval, numEpochs, trainOrValResMat )

    plot(evalInterval:evalInterval:numEpochs, trainOrValResMat.')

end

