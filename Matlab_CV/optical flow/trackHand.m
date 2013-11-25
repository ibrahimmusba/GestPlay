
      Vel = Vx.^2 + Vy.^2;
      
      maxval = max(max(Vel));
      if(maxval>4)

%             set('r','Visible','off');
            [i, j] = ind2sub(size(Vel),find(Vel == maxval)); 
            drawnow
%             r=rectangle('position',[axisIntervalx(j)-75,axisIntervaly(i)-75+30,150,120]);
%              newX = round(axisIntervalx(j));
%              newY = round(axisIntervaly(i));
            newX = round(axisIntervalx(j) + Vx(i,j));
            newY = round(axisIntervaly(i) + Vy(i,j));
%             pause(1)
            % get the velocity of this block
%             velBlock = getVel(Vel,i,j);

            count =delayCount; % set after how many count you want to take the snapshot;
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
          f2= figure
          imshow(crop)
          pos = get(f2,'position');
          set(f2,'position',[pos(1)+400,pos(2:end)]);
          end
      end