function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

numImages = size(images, 4);
imageHeight = size(images, 1);
imageWidth = size(images, 2);
imageChannels = size(images, 3);

% --------------------------------------------------------
% initialize a matrix of zeros
convolvedFeatures = zeros(numFeatures, numImages, imageHeight - patchDim + 1, imageWidth - patchDim + 1);
 WTx_bar = W*ZCAWhite*meanPatch; % prewhitening
 W= W*ZCAWhite;
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageHeight - patchDim + 1, imageWidth - patchDim + 1);
    
   
    for channel = 1:imageChannels

      % Obtain the feature (patchDim x patchDim) needed during the convolution
       feature = W(featureNum,(channel-1)*(patchDim*patchDim)+1:channel*(patchDim*patchDim));
      feature = reshape(feature,patchDim,patchDim);   
        
      % ------------------------

      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));
      % Convolve "feature" with "im", adding the result to convolvedImage
       convolvedImage = convolvedImage +conv2(im,feature,'valid'); 
      
      
      
      % ------------------------

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    convolvedImage = convolvedImage +b(featureNum) - WTx_bar(featureNum);
     
    convolvedImage = 1./(1+exp(-convolvedImage));  % compute the sigmoid
    
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end


end

