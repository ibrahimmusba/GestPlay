Vel = Vx.^2 + Vy.^2;

maxVel = max(max(Vel));
if(maxVel>velThresh) % velocity threshold
    [i, j] = ind2sub(size(Vel),find(Vel == maxVel));
    %             newX = round(axisIntervalx(j) + Vx(i,j));
    %             newY = round(axisIntervaly(i) + Vy(i,j));
    count =delayCount; % set after how many frames you want to take the snapshot;
    U = Vx;
    V = Vy;
    im = imCur;
    
    winStartI = i-winSiz;
    winEndI = i+winSiz;
    winStartJ = j-winSiz;
    winEndJ = j+winSiz;
    
    if winStartI < 1
        winStartI = 1;
        winEndI = 2*winSiz+1;
    end
    if winEndI > flowRes
        winEndI = flowRes;
        winStartI = flowRes-(2*winSiz);
    end
    if winStartJ < 1
        winStartJ = 1;
        winEndJ = 2*winSiz+1;
    end
    if winEndJ > flowRes
        winEndJ = flowRes;
        winStartJ = flowRes-(2*winSiz);
    end
    %         localVel = zeros(2*winSiz+1);
    %         localmax = 0;
    %           if (winStartI>=1 && winEndI<=flowRes) && (winStartJ>=1 && winEndJ<=flowRes)
    localVel = Vel(winStartI:winEndI,winStartJ:winEndJ);
    VEL = Vel; 
    localmax = maxVel;
    normConst = sum(sum(localVel));
    %             EntireVel = Vel;
    
    % find the centroid
    
    indX = winStartJ:winEndJ;
    indY = winStartI:winEndI;
    [X Y]= meshgrid(indX, indY);
    
    cX = sum(sum(X.*localVel))/normConst;
    cY = sum(sum(Y.*localVel))/normConst;
    
    Xl = floor(cX) ;
    Xu = ceil(cX);
    Yl = floor(cY);
    Yu = ceil(cY);
    newX = round(((axisIntervalx(Xl) + axisIntervalx(Xu))/2));
    newY = round(((axisIntervaly(Yl) + axisIntervaly(Yu))/2));
    
    
    
    %           end
end

count = count -1; % after count frames as set above, take the snapshot
waitCount = waitCount -1; % wait for 50 frames after a snapshot is taken
if count==0 && waitCount <=0;
    waitCount = snapShotsGap;
        
    % find the size of the image to be cropped
    FindCropSize;
    
    
%     display the localVel
       if(displayFullVelMap || displayLocalVelMap)
              f2= figure(2)
              if (displayLocalVelMap)
%               imagesc(localVel/max(max(localVel)));colormap gray
            
              imagesc(dilated);colormap gray
              else
              imagesc(Vel/max(max(Vel)));colormap gray
              
              end
              pos = get(f2,'position')
              set(f2,'position',[850,pos(2:end)]);
              hold on;
              plot(Xl, Yl,'r*');
       end
    % prepare to crop the image
    startx = (newX-(cropX/2));
    endx   = (newX+(cropX/2)-1);
    starty = (newY-(cropY/2));
    endy   =  (newY+(cropY/2)-1);
    
    % ensure the indices are within the limit
    if (startx>=-extraMargin && endx<=width+extraMargin) && (starty>=-extraMargin && endy<=height+extraMargin)
        if startx < 1
            startx = 1;
            endx = cropX;
        end
        if endx > width
            endx = width;
            startx = width-cropX +1;
        end
        if starty < 1
            starty = 1;
            endy = cropY;
        end
        if endy > height
            endy = height;
            starty = height-cropY+1;
        end
        
        % crop the desired portion of the hand
        %           if (startx>=1 && endx<=width) && (starty>=1 && endy<=height)
        %           uncomment the above condition to crop only if the hand is well
        %           within the image
        crop= imCur(starty:endy,startx:endx );
        colorCrop(:,:,1) = image(starty:endy,startx:endx,1);
        colorCrop(:,:,2) = image(starty:endy,startx:endx,2);
        colorCrop(:,:,3) = image(starty:endy,startx:endx,3);
        cropName = cropName+1;
        if(isImageWrite)
            imwrite(crop,[folderCropped, '\', num2str(cropName) '.jpg'],'jpg');
            imwrite(colorCrop,[folderCroppedColor, '\', num2str(cropName) '.jpg'],'jpg');
        end
        
        if(isCroppedDisplay)
            f3= figure(3)
            imshow(colorCrop)
            pos = get(f3,'position')
            set(f3,'position',[150,pos(2:end)]);
        end
        
        % Run Hand Detector
        retVal = HandDetector(crop,svmStruct,isDisplayHandDection);
%         if (retVal ==1) 
%         action= playMusic(action, player);
%         
%         end
%         
        % play music: 
        switch retVal
            case 1                
                 switch action
                    case 'play'
                        action
                        
% %                         this does the next song
%                         songNumber = mod(songNumber+1,NumSongs)
%                         if(~songNumber) 
%                             songNumber = NumSongs;
%                         end
%                                                
%                         player = audioplayer(y{songNumber}, Fs{songNumber});

                        resume(player);
                        clear action;
                        action = 'pause';
                       
                    case 'pause'
                        action
                        pause(player);
                        clear action;
                        action = 'play';
       
                 end

            case 0
              
%                 pause(1)
        end

        %           end % end statement for above if condtion
        
        
    end
    
end
