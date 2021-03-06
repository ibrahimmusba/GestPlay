function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedHeight = size(convolvedFeatures, 3);
convolvedWidth = size(convolvedFeatures, 4);

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedHeight / poolDim), floor(convolvedWidth / poolDim));
Ncols=floor(convolvedWidth / poolDim);
Nrows=floor(convolvedHeight / poolDim);
% we use mean pooling
% to use max pooling, replace 'mean' by 'max' everywhere
for imageNum = 1:numImages
  for featureNum = 1:numFeatures
     for poolRow=1:Nrows
         for poolCol=1:Ncols
           convolved = squeeze(convolvedFeatures(featureNum ,imageNum,:,:));
           poolingPatch=convolved((poolRow-1)*poolDim+1:poolRow*poolDim,(poolCol-1)*poolDim+1:poolCol*poolDim);
           pooledFeatures(featureNum, imageNum,poolRow,poolCol)= mean(mean(poolingPatch));
         end
     end
  end
end

