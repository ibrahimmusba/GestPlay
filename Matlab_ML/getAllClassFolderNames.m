function [ folder_front, folder_right, folder_left, folder_neg ] = getAllClassFolderNames( handDataSetFolder, ...
                                            croppedSize,isCentered )
%UNTITLED Given a particular gestureName this function will return folder
%names of positive examples(particular gesuture) and negative examples(all
%other images)

% neg_name{1} = '\random\randomPatches';
neg_name{1} = '\randomPatches';
neg_name{2} = '\random\randomPatches2';
neg_name{3} = '\negative';

if(isCentered)
    path ='';
else
    path ='\uncenteredAp';
end
front_name{1} = [path '\front'];
front_name{2} = [path '\front\scaled'];
        
right_name{1} = [path '\right_front'];
right_name{2} = [path '\right_back'];

left_name{1} = [path '\left_front'];
left_name{2} = [path '\left_back'];
        
for i=1:length(front_name)
    folder_front{i} = [handDataSetFolder  '\Processed' front_name{i}  '\' croppedSize]; 
end

for i=1:length(neg_name)
    folder_neg{i} = [handDataSetFolder  '\Processed' neg_name{i}  '\' croppedSize]; 
end
   
for i=1:length(right_name)
    folder_right{i} = [handDataSetFolder  '\Processed' right_name{i}  '\' croppedSize]; 
end

for i=1:length(left_name)
    folder_left{i} = [handDataSetFolder  '\Processed' left_name{i}  '\' croppedSize]; 
end 

end



