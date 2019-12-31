library(ggplot2)
library(gridExtra)
library(nnet)

total = read.csv('/Users/matthewli/Documents/Spotify Project/Spotify_PCA.csv')
print(total[total$Clusters == 0,])
print(total[order(total$Valence),])
attach(total)

# Scales
ggplot(data=total) + geom_bar(aes(x=as.factor(Mode)),stat='count', fill='darkred') + xlab('Minor/Major') + scale_x_discrete(labels=c('Minor', 'Major'))

ggplot(data=total[total$Clusters == 0,]) + geom_bar(aes(x=as.factor(Mode)),stat='count', fill='darkred') + xlab('Minor/Major') + scale_x_discrete(labels=c('Minor', 'Major'))
ggplot(data=total[total$Clusters == 1,]) + geom_bar(aes(x=as.factor(Mode)),stat='count', fill='darkred') + xlab('Minor/Major') + scale_x_discrete(labels=c('Minor', 'Major'))
ggplot(data=total[total$Clusters == 2,]) + geom_bar(aes(x=as.factor(Mode)),stat='count', fill='darkred') + xlab('Minor/Major') + scale_x_discrete(labels=c('Minor', 'Major'))

ggplot(data=total) + geom_bar(aes(x=as.factor(Key), fill=as.factor(Mode)), stat='count') + 
  scale_x_discrete(labels=c('C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B')) + scale_fill_discrete(name='Major/Minor', labels=c('Minor', 'Major')) + xlab('Key')

ggplot(data=total[total$Clusters == 0,]) + geom_bar(aes(x=as.factor(Key), fill=as.factor(Mode)), stat='count') + 
  scale_x_discrete(labels=c('C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B')) + scale_fill_discrete(name='Major/Minor', labels=c('Minor', 'Major')) + xlab('Key')
ggplot(data=total[total$Clusters == 1,]) + geom_bar(aes(x=as.factor(Key), fill=as.factor(Mode)), stat='count') + 
  scale_x_discrete(labels=c('C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B')) + scale_fill_discrete(name='Major/Minor', labels=c('Minor', 'Major')) + xlab('Key')
ggplot(data=total[total$Clusters == 2,]) + geom_bar(aes(x=as.factor(Key), fill=as.factor(Mode)), stat='count') + 
  scale_x_discrete(labels=c('C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B')) + scale_fill_discrete(name='Major/Minor', labels=c('Minor', 'Major')) + xlab('Key')

# Histograms for each cluster
overall = function(cluster, col) {
  p1 = ggplot(data=cluster, aes(x=cluster$Energy)) + geom_histogram(aes(y=..density..), color='black', fill='white') + xlab('Energy') + 
    geom_density(color = col, fill=col, alpha=0.1) + geom_vline(data=cluster, aes(xintercept=mean(cluster$Energy)), color=col, linetype="dashed")
  p2 = ggplot(data=cluster, aes(x=cluster$Liveness)) + geom_histogram(aes(y=..density..), color='black', fill='white') + xlab('Liveness') + 
    geom_density(color = col, fill=col, alpha=0.1) + geom_vline(data=cluster, aes(xintercept=mean(cluster$Liveness)), color=col, linetype="dashed")
  p3 = ggplot(data=cluster, aes(x=cluster$Speechiness)) + geom_histogram(aes(y=..density..), color='black', fill='white') + xlab('Speechiness') + 
    geom_density(color = col, fill=col, alpha=0.1) + geom_vline(data=cluster, aes(xintercept=mean(cluster$Speechiness)), color=col, linetype="dashed")
  p4 = ggplot(data=cluster, aes(x=cluster$Acousticness)) + geom_histogram(aes(y=..density..), color='black', fill='white') + xlab('Acousticness') + 
    geom_density(color = col, fill=col, alpha=0.1) + geom_vline(data=cluster, aes(xintercept=mean(cluster$Acousticness)), color=col, linetype="dashed")
  p5 = ggplot(data=cluster, aes(x=cluster$Danceability)) + geom_histogram(aes(y=..density..), color='black', fill='white') + xlab('Danceability') + 
    geom_density(color = col, fill=col, alpha=0.1) + geom_vline(data=cluster, aes(xintercept=mean(cluster$Danceability)), color=col, linetype="dashed")
  p6 = ggplot(data=cluster, aes(x=cluster$Valence)) + geom_histogram(aes(y=..density..), color='black', fill='white') + xlab('Valence') + 
    geom_density(color = col, fill=col, alpha=0.1) + geom_vline(data=cluster, aes(xintercept=mean(cluster$Valence)), color=col, linetype="dashed")
  grid.arrange(p1, p2, p3, p4, p5, p6, ncol=3, nrow=2)
}
overall(total, 'turquoise')
overall(total[Clusters == 0,], 'orange')
overall(total[Clusters == 1,], 'pink')
overall(total[Clusters == 2,], 'purple')

# Means
means = aggregate(total, by = list(Clusters), mean)
print(means)

# Histograms/Densities for each Feature
density = function(feature, mean, label) {
  ggplot(total) + geom_density(aes(x=feature, color = as.factor(Clusters), fill=as.factor(Clusters)), alpha=0.25) + 
    scale_fill_discrete(name='Clusters') + scale_color_discrete(guide='none') + scale_alpha(guide='none') + xlab(label) + 
    geom_vline(data=means, aes(xintercept=mean, color=as.factor(Clusters)), linetype="dashed")
}
density(Energy, means$Energy, 'Energy')
density(Liveness, means$Liveness, 'Liveness')
density(Speechiness, means$Speechiness, 'Speechiness')
density(Acousticness, means$Acousticness, 'Acousticness')
density(Danceability, means$Danceability, 'Danceability')
density(Valence, means$Valence, 'Valence')

histogram = function(feature, label) {
  ggplot(total) + geom_histogram(aes(x=feature, color = as.factor(Clusters), fill=as.factor(Clusters)), alpha=0.25) + 
    scale_fill_discrete(name='Clusters') + scale_color_discrete(guide='none') + scale_alpha(guide='none') + xlab(label)
}

histogram(Energy, 'Energy')
histogram(Liveness, 'Liveness')
histogram(Speechiness, 'Speechiness')
histogram(Acousticness, 'Acousticness')
histogram(Danceability, 'Danceability')
histogram(Valence, 'Valence')

# Boxplots for each feature

box = function(feature, label) {
  ggplot(total) + geom_boxplot(aes(x=as.factor(Clusters), y=feature, color=as.factor(Clusters), fill=as.factor(Clusters)), alpha=0.25) +
    scale_fill_discrete(name='Clusters') + scale_color_discrete(guide='none') + scale_alpha(guide='none') + ylab(label) + xlab('Clusters')
}
box(Energy, 'Energy')
box(Liveness, 'Liveness')
box(Speechiness, 'Speechiness')
box(Acousticness, 'Acousticness')
box(Danceability, 'Danceability')
box(Valence, 'Valence')

# Anova
energyanova = aov(Energy~as.factor(Clusters))
summary(energyanova)
qqnorm(energyanova$residuals)
TukeyHSD(energyanova)
pairwise.t.test(Energy, Clusters, p.adj='bonferroni')

livenessanova = aov(Liveness~as.factor(Clusters))
summary(livenessanova)
qqnorm(livenessanova$residuals)
TukeyHSD(livenessanova)
pairwise.t.test(Liveness, Clusters, p.adj='bonferroni')

speechinessanova = aov(Speechiness~as.factor(Clusters))
summary(speechinessanova)
qqnorm(speechinessanova$residuals)
TukeyHSD(speechinessanova)
pairwise.t.test(Speechiness, Clusters, p.adj='bonferroni')

acousticnessanova = aov(Acousticness~as.factor(Clusters))
summary(acousticnessanova)
qqnorm(acousticnessanova$residuals)
TukeyHSD(acousticnessanova)
pairwise.t.test(Acousticness, Clusters, p.adj='bonferroni')

danceabilityanova = aov(Danceability~as.factor(Clusters))
summary(danceabilityanova)
qqnorm(danceabilityanova$residuals)
TukeyHSD(danceabilityanova)
pairwise.t.test(Danceability, Clusters, p.adj='bonferroni')

valenceanova = aov(Valence~as.factor(Clusters))
summary(valenceanova)
qqnorm(valenceanova$residuals)
TukeyHSD(valenceanova)
pairwise.t.test(Valence, Clusters, p.adj='bonferroni')

# Predicting
multi = multinom(Clusters~Energy+Liveness+Speechiness+Acousticness+Danceability+Valence)
summary(multi)

top = read.csv('/Users/matthewli/Documents/Spotify Project/Top2018.csv')
predicted = c()
for (x in 1:nrow(top)) {
  x = predict(multi, data.frame(Energy=top[x,'Energy'], Liveness=top[x,'Liveness'], Speechiness=top[x,'Speechiness'], Acousticness=top[x,'Acousticness'], Danceability=top[x,'Danceability'], Valence=top[x,'Valence']), type='probs')
  predicted = c(predicted, names(x[x==max(x)]))
}
print(predicted)
top$Predict = predicted
print(top[,c(2,3,18)])
