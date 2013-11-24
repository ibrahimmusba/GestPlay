%% setup the camera 

vid.camID = camID; % default value is 1

vidSource = lower(vidSource);

if strcmpi(vidSource, 'camera')
 % if kindOfMovie is camera
        vid.bUseCam = 1; %matlab built in
        vid.Height = height;   vid.Width  = width;

        try  
            imaqreset; % resets the image aquisition block if it is present
        catch 
            fprintf('\nNo image aquisition block! \n Looking for videoinput library(this is a windows ONLY library)... ');
            vid.bUseCam = 2; %videoInput
            try
                VI = vi_create();
                vi_delete(VI);
                fprintf('FOUND IT!\n');
            catch %#ok<CTCH>
                error('no library available for camera input');
            end
        end
        
        if(vid.bUseCam==1) %matlab built in

    %     get info on supported formats for this capture device
            camInfo = imaqhwinfo('winvideo',vid.camID);
            supportedVid = camInfo.SupportedFormats; % it stores the formats supported 
    
        %     supportedVid contains all the supported formats as strings , for example 
            %     'YUV2_320x240' is one such format. 
                 
            splitStr = regexpi(supportedVid,'x|_','split');
            pickedFormat = 0;
            resolutionFormat = Inf;
            for ik = 1:length(supportedVid)
                resW = str2double(splitStr{ik}{2});
                resH = str2double(splitStr{ik}{3});
                if (resW > (vid.Width-1) )&&(resH > (vid.Height-1) )&& (resW*resH)<resolutionFormat
                    resolutionFormat = (resW*resH);
                    pickedFormat = ik;
                end
            end

%             pickedFormat =  4; %Apexit
%              resolutionFormat = 640*480;%Apexit
            % pick the selected format, color and a region of interest:
            vid.camIn = videoinput('winvideo',vid.camID,supportedVid{pickedFormat});
%             set(vid.camIn, 'ReturnedColorSpace', 'gray');
            set(vid.camIn, 'ReturnedColorSpace', 'rgb');

             set(vid.camIn, 'ROIPosition', [0 0 vid.Width vid.Height]);
            %let the video go on forever, grab one frame every update time, maximum framerate:
            triggerconfig(vid.camIn, 'manual');
            src = getselectedsource(vid.camIn);
            if isfield(get(src), 'FrameRate')
                frameRates = set(src, 'FrameRate');
                src.FrameRate = frameRates{1};    
            end
        
            start(vid.camIn);
            pause(0.001);
       

        else %if vid.bUseCam == 2 %using videoinput library
            vid.camIn = vi_create();
            numDevices = vi_list_devices(vid.camIn);
            if numDevices<1,    error('video input found no cameras');end
            vi_setup_deviceProper(vid.camIn, camID-1, vid.Height, vid.Width, 30);
        end 
    
       
else         %we open the file for reading
       try
           vid = VideoReader(vidSource);
       catch
           fprintf('\n please check the file name!\n');
       end
end
