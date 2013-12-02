
clear all
close all

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


winSiz = 10; % decides how many flow vectors we want to consider to find the centroid

if 2*winSiz+1 >flowRes
    error 'reduce window size';
end
% indY = -winSize:winSiz;
% indX = -winSize:winSiz;

cropX = 120; % size of the bounding box
cropY = 150;
extraMargin = 30; % this parameter defines how much of hand could be outside
camID = 1 ; % default value is 1
vidSource = 'camera'; % selects the camera as source
% vidSource = 'LipVid.avi'; % selects the video file


count =-1;
waitCount = -1;
delayCount =2 ; % set after how many count you want to take the snapshot;
snapShotsGap = 7; % set after how many frames you want to take another snapshot
% open the video
openVideo; % sets the video parameters and open a video object

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