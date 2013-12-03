function [H_train Y_train H_test Y_test] = loadHandHoGData(folder_pos, folder_neg)
%folder_pos and folder_neg are folder names which contain all the image
%files. The images should all be of the same dimension. folder_neg can be
%an array of folder names having negative images

H = []; %Feature Vectors
X = []; %Image Vectors
Y = []; %Output Labels

n_total = 0;

%Read all positive examples
fprintf('Reading all Positive Examples and calculating HoG... \n')
for i = 1:length(folder_pos)
    posImageFiles = dir([folder_pos{i} '\' '*.jpg']); 
    for k = 1:length(posImageFiles)
        filename = posImageFiles(k).name;
        img = imread([folder_pos{i} '\' filename]);
        H = [H, HoG(img)];
        if (size(img,3) == 3)
            img = rgb2gray(img);
        end
        X = [X, img(:)];
    end
    Y = [Y, ones(1,length(posImageFiles))];
    n_total = n_total + length(posImageFiles);
end

%Read all negative examples
fprintf('Reading all Negative Examples and calculating HoG... \n')
for i = 1:length(folder_neg)
    negImageFiles = dir([folder_neg{i} '\' '*.jpg']); 
    for k = 1:length(negImageFiles)
        filename = negImageFiles(k).name;
        img = imread([folder_neg{i} '\' filename]);        
        H = [H, HoG(img)];
        if (size(img,3) == 3)
            img = rgb2gray(img);
        end
        X = [X, img(:)];
    end
    Y = [Y, -1.*ones(1,length(negImageFiles))];
    n_total = n_total + length(negImageFiles);
end


%Shuffle training data
randInd = randperm(n_total);

H = H(:,randInd);
X = X(:,randInd);
Y = Y(:,randInd);

%normalize the data from 0 to 1
minH = min(min(H));
maxH = max(max(H))
H = (H-minH)./(maxH-minH);

%Whiten the data
%hmean = mean(H,2);
%H_norm = H - hmean*ones(1,n_total);
%X = X - immean*ones(1,n_total);

%cov = 1/n_total * (H*H');

%H = real(inv(real(sqrtm(cov))))*H;



n_train = round(n_total*2/3);

H_train = H(:,1:n_train);
Y_train = Y(1:n_train);

H_test = H(:, n_train+1:end);
Y_test = Y(n_train+1:end);

end

