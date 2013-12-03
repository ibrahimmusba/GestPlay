se = strel('disk',1);
% dilated = imdilate(VEL>1,se);
dilated = imopen(VEL/localmax,se);