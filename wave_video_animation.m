
% declare your variables. t = time, w = raidial freq, b =wave number,         
% z = distance traveled, e = exponential term e raised to the - a * z,
% a = propogation constant alpha, e_field = electric field functin

w = 2*pi*100*10^6
a = 0.086
b = 2.57
z = [0:0.05:20]
t = 0:0.25*10^-9:100*10^-9
e = exp(-a*z)
e_field= 5*e.*cos(w*t - b*z)

% for loop will continuously plot the graph as both a function of space and
% time. Get frame function will capture each frame as it passes through the
% loop

for i=1:length(t)
    grid on
    title(['E Field strength at time t=' num2str(t(i)) 'seconds'])
    xlabel("Position [m]")
    ylabel("Field Strength [V/m]")
    hold on
    plot(z(1:i),e_fieldf(1:i))
    ylim([-6 6])
    Frame(i) = getframe(gcf)
    pause(.2)
    
    
end


%videowriter function saves the video as a .avi file 

v= VideoWriter('electric field sim', 'Uncompressed AVI')
open(v)
writeVideo(v,Frame)
close(v)













