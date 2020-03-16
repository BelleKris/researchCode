###(1)The regulation of miRNA makers in cancer-related pathways
pathway_file=list.files("refine_pathway_marker")
pathway_dir= paste("./refine_pathway_marker/",pathway_file,sep="")
n =length(pathway_dir)   
pathway_name=gsub(" marker.txt","",pathway_file)

marker_file=list.files("miRNA_marker_target")
marker_dir= paste("./miRNA_marker_target/",marker_file,sep="")
m =length(marker_dir)   
cancer_name=gsub("_miRNA_target.txt","",marker_file)
count=c()
for (i in 1:n){
for (j in 1:m){
pathway = read.table(file = pathway_dir[i], sep="\t",quote="",header=T) 
cancer_marker=read.table(file = marker_dir[j], sep="\t",quote="",header=T) 
nn=which(as.matrix(cancer_marker[,2]) %in% as.matrix(pathway[,1]))
marker_nn=length(unique(cancer_marker[nn,1]))
marker_N=length(unique(cancer_marker[,1]))
count1=cbind(pathway_name[i],cancer_name[j],marker_nn,marker_N,marker_nn/marker_N)
count=rbind(count,count1)
pathway_marker=cancer_marker[nn,2]
write.table(unique(pathway_marker),paste(paste("refine_pathway_cancer_miRNA/",paste(pathway_name[i],cancer_name[j],sep="_"),sep=""),sep = ""," marker.txt"),sep="\t",quote=F,row.names=F)
}
}
colnames(count)=c("pathway","cancer","marker_nn","marker_N","marker_percent")
write.table(count,"refine_pathway_cancer_miRNA_count_percent.txt",sep="\t",quote=F,row.names=F)


###(2)The regulation of miRNA makers in the genes of cancer-related pathways
pathway_file=list.files("refine_pathway_marker")
pathway_dir= paste("./refine_pathway_marker/",pathway_file,sep="")
n =length(pathway_dir)   
pathway_name=gsub(" marker.txt","",pathway_file)

marker_file=list.files("miRNA_marker_target")
marker_dir= paste("./miRNA_marker_target/",marker_file,sep="")
m =length(marker_dir)   
cancer_name=gsub("_miRNA_target.txt","",marker_file)

for (i in 1:n){
pathway = read.table(file = pathway_dir[i], sep="\t",quote="",header=T) 
count=c()
for (j in 1:m){
cancer_marker=read.table(file = marker_dir[j], sep="\t",quote="",header=T)
for(k in 1: nrow(pathway)){
nn=which(as.matrix(cancer_marker[,2])==pathway[k,1])
marker_nn=length(unique(cancer_marker[nn,1]))
marker_N=length(unique(cancer_marker[,1]))
count1=cbind(pathway_name[i],cancer_name[j],as.character(pathway[k,1]),marker_nn,marker_N,marker_nn/marker_N)
count=rbind(count,count1)
}
}
colnames(count)=c("pathway_name","cancer_name","pathway_gene","marker_map_n","marker_N","map_percent")
write.table(count,paste(paste("refine_pathway_gene_cancer_marker/",pathway_name[i],sep=""),sep = "","_gene_marker.txt"),sep="\t",quote=F,row.names=F)

}

