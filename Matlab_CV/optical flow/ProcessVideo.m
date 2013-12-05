
clear all
close all

%% set parameters to display/write the cropped image

isDisplayVideo = 1; % Do NOT change .display the video
isDisplayVelVector = 1; % to display the velocity vectors
isDisplayHandDection = 0; % to display the hand detection



isCroppedDisplay = 0 ;  % set to 0 if dont want to display
isImageWrite = 0 ;      % set to 0 if dont want to write

displayFullVelMap = 0;  % set to 0 if dont want to display full vel map
displayLocalVelMap= 0;  % set to 0 if dont want to display local vel map
% when both local and full vel map is enabled, it displays only local vel
%dataset_folder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset';
dataset_folder = 'C:\Users\Apexit\Dropbox\courses\fall-13\fall-13\442\project\Papers_Project_442_545\Hand Database\dataset';

display('loading the classifier....')
% load([dataset_folder '\LearnedData\SVM_front_p2.mat']);
load([dataset_folder '\LearnedData\SVM_front_p2_scaled.mat']);
% load([dataset_folder '\LearnedData\SVM_front_p2_scaled_othernegative.mat']);

addpath ../
addpath soundGui

%% set camera parameters
camInfo = imaqhwinfo('winvideo',1);
supportedVid = camInfo.SupportedFormats;
% chose the width and height based on above information. Don't keep it in
% too high resolution
width = 320; 
height = 240;

% flow parameter
flowRes = 30 ; %flow resolution 
scale =8; %to scale the vectors by desired amount


velThresh = 3;
winSiz = 5; % decides how many flow vectors we want to consider to find the centroid

if 2*winSiz+1 >flowRes
    error 'reduce window size';
end
% indY = -winSize:winSiz;
% indX = -winSize:winSiz;

%% set the cropped image size
cropX = 120; % size of the bounding box
cropY = 168;
% cropX = 88; % size of the bounding box
% cropY = 112;


extraMargin = 30; % this parameter defines how much of hand could be outside
camID = 1 ; % default value is 1
vidSource = 'camera'; % selects the camera as source
% vidSource = 'LipVid.avi'; % selects the video file

%% music player setting
songPath = 'C:\Users\Apexit\Dropbox\courses\fall-13\fall-13\442\project\Papers_Project_442_545\Hand Database\dataset\songs\';

songName = [songPath 'kolaveri.mp3'];
display('loading soundfile...')
[y,Fs,NBITS,OPTS] = readMp3(songName,[400000,5000000]);
player = audioplayer(y, Fs);
action = 'play';
play(player);
pause(player);

%% set paramters for frame lag 
count =-1;
waitCount = -1;

% delayCount =floor(velThresh/2) ; % set after how many count you want to take the snapshot;
delayCount =max(ceil(velThresh/2),2) ; % set after how many count you want to take the snapshot;
snapShotsGap = 4; % set after how many frames you want to take another snapshot
% open the video
openVideo; % sets the video parameters and open a video object

% create a folder to save the cropped image
if(isImageWrite)
    path = '';
    folderCropped = [path 'Cropped'];
    folderCroppedColor = [path 'CroppedColor'];

    mkdir(folderCropped);
    mkdir(folderCroppedColor);
    croppedFiles = dir([folderCropped, '\','*.jpg']);
    cropName = length(croppedFiles);
end

% the first frame! 
frameNum=1; % time index for frames
image = fetchFrame(vid, frameNum, vidSource);
imCur = rgb2gray(image);
[Ix Iy It] = findDerivatives(imCur,frameNum);
% axis for vel vector
    axisInterval = linspace(1,width,flowRes+2);
    axisIntervalx = axisInterval(2:end-1) ;
    axisInterval = linspace(1,height,flowRes+2);
    axisIntervaly = axisInterval(2:end-1) ;

% display the frame
% if(isDisplayVideo)
    figure('NumberTitle','off');
    handleImage = imagesc(imCur); colormap(gray);
    hold on;
    if(isDisplayVelVector)
        handleQuiver = quiver(axisIntervalx,axisIntervaly, zeros(flowRes),  zeros(flowRes),0 ,'m','MaxHeadSize',5,'Color',[.9 .2 .1]);%, 'LineWidth', 1);
    end
    axis image;
    fig = gcf; 
% end
% r= rectangle('position',[0 0 width, height]);

display('GestPlay starts here...')
% process rest of the video
while(1) 
    frameNum = frameNum+1; % jump to the next frame
    
    image = fetchFrame(vid, frameNum, vidSource);
    imCur = rgb2gray(image); 
    [Ix Iy It] = findDerivatives(imCur,frameNum);
    [Vx Vy] = findMotion(Ix ,Iy ,It,flowRes);
    
    % display the current frame and vectors
    
        if (ishandle(fig))
         set(handleImage ,'cdata',image);
         if(isDisplayVelVector)
            set(handleQuiver ,'UData', scale*Vx, 'VData', scale*Vy);
         end
        else
            break;
        end
    
    trackHand;        
           
%     figure(2)
%     imagesc(Vx.^2 + Vy.^2)
%     trackMaxVel;
    
    pause(0.0001);
    
end
   
pause(player)
close all

% close the camera and clear the memory 
      
if strcmpi(vidSource,'camera')
  if vid.bUseCam ==2
      vi_stop_device(vid.camIn, vid.camID-1);
      vi_delete(vid.camIn);
  else
      delete(vid.camIn); 
  end
else
    delete(vid); 
end 
