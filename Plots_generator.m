load Pegasos_01_1
subplot(2,1,1)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('frequency vector, k=1')
xlabel('number of iterations')
ylabel('accuracy')

clear
load Pegasos_01_1_B
subplot(2,1,2)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('binary vector, k=1')
xlabel('number of iterations')
ylabel('accuracy')

%% k=100
figure
load Pegasos_01_100
subplot(2,1,1)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('frequency vector, k=100')
xlabel('number of iterations')
ylabel('accuracy')

clear
load Pegasos_01_100_B
subplot(2,1,2)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('binary vector, k=100')
xlabel('number of iterations')
ylabel('accuracy')

%% k=1000
figure
load Pegasos_01_1000
subplot(2,1,1)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('frequency vector, k=1000')
xlabel('number of iterations')
ylabel('accuracy')

clear
load Pegasos_01_1000_B
subplot(2,1,2)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('binary vector, k=1000')
xlabel('number of iterations')
ylabel('accuracy')

%% full
figure
load Pegasos_01_full
subplot(2,1,1)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('frequency vector, k=N')
xlabel('number of iterations')
ylabel('accuracy')

clear
load Pegasos_01_full_B
subplot(2,1,2)
plot(1:size(TrajectoryAll,2),TrajectoryAll)
title('binary vector, k=N')
xlabel('number of iterations')
ylabel('accuracy')
