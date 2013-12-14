function [ folder_pos folder_neg ] = getBinaryClassFolderNames( handDataSetFolder, ...
                                            gestureName, croppedSize )
%UNTITLED Given a particular gestureName this function will return folder
%names of positive examples(particular gesuture) and negative examples(all
%other images)

%  neg_name{1} = '\random\randomPatches';
neg_name{1} = '\randomPatches';
neg_name{2} = '\random\randomPatches2';
neg_name{3} = '\negative';


switch gestureName
    case 'front'
        pos_name{1} = '\front';
        pos_name{2} = '\front\scaled';
        
        neg_name{4} = '\right_front';
        neg_name{5} = '\right_back';
        neg_name{6} = '\left_front';
        neg_name{7} = '\left_back';
        
    case 'right'
        pos_name{1} = '\right_front';
        pos_name{2} = '\right_back';
        
        neg_name{4} = '\front';
        neg_name{5} = '\front\scaled';
        neg_name{6} = '\left_front';
        neg_name{7} = '\left_back';

    case 'left'
        pos_name{1} = '\left_front';
        pos_name{2} = '\left_back';
        
        neg_name{4} = '\front';
        neg_name{5} = '\front\scaled';
        neg_name{6} = '\right_front';
        neg_name{7} = '\right_back';
        
    case 'uncentered'
         pos_name{1} = '\uncenteredAp\front';
        
        neg_name{4} = '\right_front';
        neg_name{5} = '\right_back';
        neg_name{6} = '\left_front';
        neg_name{7} = '\left_back';
        
    otherwise
        error('error in getFolderNames: gestureName not defined');
        
end

for i=1:length(pos_name)
    folder_pos{i} = [handDataSetFolder  '\Processed' pos_name{i}  '\' croppedSize]; 
end

for i=1:length(neg_name)
    folder_neg{i} = [handDataSetFolder  '\Processed' neg_name{i}  '\' croppedSize]; 
end
    
end

