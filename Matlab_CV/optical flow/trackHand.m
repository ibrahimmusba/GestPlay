
      Vel = Vx.^2 + Vy.^2;
      
      maxVel = max(max(Vel));
      if(maxVel>4) % velocity threshold 
            [i, j] = ind2sub(size(Vel),find(Vel == maxVel)); 
            newX = round(axisIntervalx(j) + Vx(i,j));
            newY = round(axisIntervaly(i) + Vy(i,j));
            count =delayCount; % set after how many frames you want to take the snapshot;
       U = Vx;
       V = Vy;
       im = imCur;
            winSiz = 7;
            winStartI = i-winSiz;
            winEndI = i+winSiz;
            winStartJ = j-winSiz;
            winEndJ = j+winSiz;
%           if winStartX < 1
%               winStartX = 1;
%               winEndX = 2*winSiz+1;
%           end
%           if winEndX > flowRes
%              winEndX = flowRes;
%              winStartX = flowRes-(2*winSiz+1);
%           end
%           if winStartY < 1
%             winStartY = 1;
%             endy = 2*winSiz+1;
%           end
%           if winEndY > flowRes
%               winEndY = flowRes;
%              winStartY = flowRes-(2*winSiz+1);
%           end
        localVel = zeros(2*winSiz+1);
        localmax = 0;
          if (winStartI>=1 && winEndI<=flowRes) && (winStartJ>=1 && winEndJ<=flowRes)
            localVel = Vel(winStartI:winEndI,winStartJ:winEndJ);
            localmax = maxVel;
            EntireVel = Vel;
          end
      end
      
      count = count -1; % after count frames as set above, take the snapshot
      waitCount = waitCount -1; % wait for 50 frames after a snapshot is taken
      if count==0 && waitCount <=0;
          waitCount = snapShotsGap;
          startx = (newX-(cropX/2));
          endx   = (newX+(cropX/2));
          starty = (newY-(cropY/2));
          endy   =  (newY+(cropY/2));
          
%           if startx < 1
%               startx = 1;
%               endx = cropX;
%           end
%           if endx > width
%              endx = width;
%              startx = width-cropX;
%           end
%           if starty < 1
%             starty = 1;
%             endy = cropY;
%           end
%           if endy > height
%               endy = height;
%              starty = height-cropY;
%           end
          if (startx>=1 && endx<=width) && (starty>=1 && endy<=height)
          crop= imCur(starty:endy,startx:endx );
          f2= figure(2)
          imagesc(localVel);colormap gray
          pos = get(f2,'position')
          set(f2,'position',[950,pos(2:end)]);
 
          f3= figure(3) 
          imshow(crop)
          pos = get(f3,'position')
          set(f3,'position',[150,pos(2:end)]);
                  
          
          end
      end