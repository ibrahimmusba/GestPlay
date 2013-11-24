function [frame] = fetchFrame(vid, frameNum, vidSource)
%this function returns the frame number as specified by frameNum  and based
%on the type of source

vidSource = lower(vidSource);
if( strcmp(vidSource,'camera') )
     if vid.bUseCam==2 %videoinput lib:
        vi_is_frame_new(vid.camIn, vid.camID-1);
        frame  = vi_get_pixelsProper(vid.camIn, vid.camID-1,vid.Height,vid.Width);
    else %image aqcuisition toolbox:
        frame = flipdim(squeeze(getsnapshot(vid.camIn)),2);
    end
else % if it is a video file stored disk
    frameNum = mod(frameNum, vid.NumberOfFrames) + 1; % repeats the video infinite times
    frame = rgb2gray(read(vid,frameNum));
end
    
    
