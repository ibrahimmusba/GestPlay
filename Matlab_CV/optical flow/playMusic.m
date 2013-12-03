% playMusic
function action = playMusic( action,player)


    switch action
        case 'play'
            action
            resume(player);
            clear action;
            action = 'pause';
            pause(1)
            return;
        case 'pause'
            action
            pause(player);
            clear action;
            action = 'play';
            pause(1)
            return
    end
end