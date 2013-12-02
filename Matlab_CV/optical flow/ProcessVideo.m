
clear all
close all

%% set parameters to display/write the cropped image
isCroppedDisplay = 1 ;  % set to 0 if dont want to display
isImageWrite = 0 ;      % set to 0 if dont want to write

displayFullVelMap = 0;  % set to 0 if dont want to display full vel map
displayLocalVelMap= 0;  % set to 0 if dont want to display local vel map
% when both local and full vel map is enabled, it displays only local vel

%% set camera parameters
camInfo = imaqhwinfo('winvideo',1);
supportedVid = camInfo.SupportedFormats
% chose the width and height based on above information. Don't keep it in
% too high resolution
width = 320; 
height = 240;

% flow parameter
flowRes = 30 ; %flow resolution 
scale =8; %to scale the vectors by desired amount


velThresh = 6;
winSiz = 5; % decides how many flow vectors we want to consider to find the centroid

if 2*winSiz+1 >flowRes
    error 'reduce window size';
end
% indY = -winSize:winSiz;
% indX = -winSize:winSiz;

%% set the cropped image size
cropX = 120; % size of the bounding box
cropY = 144;
% cropX = 88; % size of the bounding box
% cropY = 112;


extraMargin = 30; % this parameter defines how much of hand could be outside
camID = 1 ; % default value is 1
vidSource = 'camera'; % selects the camera as source
% vidSource = 'LipVid.avi'; % selects the video file

%% set paramters for frame lag 
count =-1;
waitCount = -1;
delayCount =floor(velThresh/2) ; % set after how many count you want to take the snapshot;
snapShotsGap = 4; % set after how many frames you want to take another snapshot
% open the video
openVideo; % sets the video parameters and open a video object

% create a folder to save the cropped image
path = '';
folderCropped = [path 'Cropped'];
folderCroppedColor = [path 'CroppedColor'];

mkdir(folderCropped);
mkdir(folderCroppedColor);
croppedFiles = dir([folderCropped, '\','*.jpg']);
cropName = length(croppedFiles);
% the first frame! 
frameNum=1; % time index for frames
image = fetchFrame(vid, frameNum, vidSource);
imCur = rgb2gray(image);
[Ix Iy It] = findDerivatives(imCur,frameNum);

% display the frame
figure('NumberTitle','off');
handleImage = imagesc(imCur); colormap(gray);
hold on;
axisInterval = linspace(1,width,flowRes+2);
axisIntervalx = axisInterval(2:end-1) ;
axisInterval = linspace(1,height,flowRes+2);
axisIntervaly = axisInterval(2:end-1) ;

handleQuiver = quiver(axisIntervalx,axisIntervaly, zeros(flowRes),  zeros(flowRes),0 ,'m','MaxHeadSize',5,'Color',[.9 .2 .1]);%, 'LineWidth', 1);
axis image;
fig = gcf; 
% r= rectangle('position',[0 0 width, height]);

dataset_folder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset';
load([dataset_folder '\LearnedData\SVMstruct_p2.mat']);
% process rest of the video
while(1) 
    frameNum = frameNum+1; % jump to the next frame
    
    image = fetchFrame(vid, frameNum, vidSource);
    imCur = rgb2gray(image); 
    [Ix Iy It] = findDerivatives(imCur,frameNum);
    [Vx Vy] = findMotion(Ix ,Iy ,It,flowRes);
    
    % display the current frame and vectors
    if ishandle(fig)
     set(handleImage ,'cdata',image);
     set(handleQuiver ,'UData', scale*Vx, 'VData', scale*Vy);
    else
        break;
    end
    
    trackHand;        
           
%     figure(2)
%     imagesc(Vx.^2 + Vy.^2)
%     trackMaxVel;
    
    pause(0.0001);
    
end
   

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
