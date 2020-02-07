display("thank you for taking the time to run my code, please respond to the prompts :)")


%prompting user to enter values for freq, conductivity, relative
%permitivity

freq = input("enter frequency ")
con = input("enter conductivity ")
er = input("enter value for Er ")

% declaring variables for a streamlined formula later 

w = freq/(2*pi)
E = er*8.85*10^-12
u = 4*pi*10^-7


%Calculating propogation constants alpha and beta for general conductive
%media

alpha = (w*sqrt(u*E)/sqrt(2))*sqrt((sqrt(1+(con/(u*E))^2)-1))

beta = (w*sqrt(u*E)/sqrt(2))*sqrt((sqrt(1+(con/(u*E))^2)+1))