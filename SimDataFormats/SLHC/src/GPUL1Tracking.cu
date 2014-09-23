// for tracklet finding
// blockidx.x will be the sector
// blockIdx.y will be the layer

// dimensions:
// gridDim.x = number of sectors
// gridDim.y = number of layers -1
// gridDim.z = 3 (area directly on the next layer, same sector and the other two areas
//                 for sectorId +1 or -1)


// blockDim.x = stubs in sector
#include "math_constants.h" // needed for pi



__host__ __device__
inline
double r(StubsSoAElement* element, int id)
{
	return sqrt(element->x[id]*element->x[id] + element->y[id]*element->y[id]);
}

__host__ __device__
inline
double r(GPUL1TTrack& track, int id)
{
	return sqrt(track.seed.x[id]*track.seed.x[id] + track.seed.y[id]*track.seed.y[id]);
}

__host__ __device__
inline
double phi(GPUL1TTrack& track, int id)
{
	return atan2(track.seed.y[id],track.seed.x[id]);
}

__host__ __device__
inline
double phi(StubsSoAElement* element, int id)
{

	return atan2(element->y[id], element->x[id]);

}
__host__ __device__
void invert(double M[4][8], int n){

	//assert(n<=4);

	unsigned int i,j,k;
	double ratio,a;

	for(i = 0; i < n; i++){
		for(j = n; j < 2*n; j++){
			if(i==(j-n))
				M[i][j] = 1.0;
			else
				M[i][j] = 0.0;
		}
	}

	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			if(i!=j){
				ratio = M[j][i]/M[i][i];
				for(k = 0; k < 2*n; k++){
					M[j][k] -= ratio * M[i][k];
				}
			}
		}
	}

	for(i = 0; i < n; i++){
		a = M[i][i];
		for(j = 0; j < 2*n; j++){
			M[i][j] /= a;
		}
	}
}


__device__
void calculateDerivatives(GPUL1TTrack& track){

	int numLayers = 6;

	int n = track.seed.nStubs;gmai
	for(int i = 0; i < n; ++i)
	{


			double ri = r(track,i);
			double zi = track.seed.z[i];

			double sigmax = track.seed.sigmax[i];
			double sigmaz = track.seed.sigmaz[i];


			double rinv = track.rinv;
			//cout << "i layer "<<i<<" "<<stubs_[i].layer()<<endl;

			//	if (stubs_[i].layer()<1000){
			//here we handle a barrel hit

			//first we have the phi position
			track.D[0][4*i]=-0.5*ri*ri/sqrt(1-ri*ri*rinv*rinv/4)/sigmax;
			track.D[1][4*i]=ri/sigmax;
			track.D[2][4*i]=0.0;
			track.D[3][4*i]=0.0;

			//second the z position
			track.D[0][4*i+1]=0.0;
			track.D[1][4*i]=0.0;
			track.D[2][4*i]=(2/rinv)*asin(0.5*ri*rinv)/sigmaz;
			track.D[3][4*i]=1.0/sigmaz;

			//			}
			//			//    else {
			//			//	//here we handle a disk hit
			//			//	//first we have the r position
			//			//
			//			//	double r_track=2.0*sin(0.5*rinv_*(zi-z0_)/t_)/rinv_;
			//			//	double phi_track=phi0_-0.5*rinv_*(zi-z0_)/t_;
			//			//
			//			//	int iphi=stubs_[i].iphi();
			//			//	double phii=stubs_[i].phi();
			//			//
			//			//	double width=4.608;
			//			//	double nstrip=508.0;
			//			//	if (ri<60.0) {
			//			//	  width=4.8;
			//			//	  nstrip=480;
			//			//	}
			//			double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
			//
			//			if (stubs_[i].z()>0.0) Deltai=-Deltai;
			//			double theta0=asin(Deltai/ri);
			//
			//			double rmultiplier=-sin(theta0-(phi_track-phii));
			//			double phimultiplier=r_track*cos(theta0-(phi_track-phii));
			//
			//
			//			double drdrinv=-2.0*sin(0.5*rinv_*(zi-z0_)/t_)/(rinv_*rinv_)
			//									+(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(rinv_*t_);
			//			double drdphi0=0;
			//			double drdt=-(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(t_*t_);
			//			double drdz0=-cos(0.5*rinv_*(zi-z0_)/t_)/t_;
			//
			//			double dphidrinv=-0.5*(zi-z0_)/t_;
			//			double dphidphi0=1.0;
			//			double dphidt=0.5*rinv_*(zi-z0_)/(t_*t_);
			//			double dphidz0=0.5*rinv_/t_;
			//
			//			D_[0][j]=drdrinv/sigmaz;
			//			D_[1][j]=drdphi0/sigmaz;
			//			D_[2][j]=drdt/sigmaz;
			//			D_[3][j]=drdz0/sigmaz;
			//			j++;
			//			//second the rphi position
			//			D_[0][j]=(phimultiplier*dphidrinv+rmultiplier*drdrinv)/sigmax;
			//			D_[1][j]=(phimultiplier*dphidphi0+rmultiplier*drdphi0)/sigmax;
			//			D_[2][j]=(phimultiplier*dphidt+rmultiplier*drdt)/sigmax;
			//			D_[3][j]=(phimultiplier*dphidz0+rmultiplier*drdz0)/sigmax;
			//			j++;
			//		}

	}

	//cout << "Exact rinv derivative: "<<i<<" "<<D_[0][j-2]<<" "<<D_[0][j-1]<<endl;
	//cout << "Exact phi0 derivative: "<<i<<" "<<D_[1][j-2]<<" "<<D_[1][j-1]<<endl;
	//cout << "Exact t derivative   : "<<i<<" "<<D_[2][j-2]<<" "<<D_[2][j-1]<<endl;
	//cout << "Exact z0 derivative  : "<<i<<" "<<D_[3][j-2]<<" "<<D_[3][j-1]<<endl;




	//cout << "D:"<<endl;
	//for(unsigned int j=0;j<2*n;j++){
	//  cout <<D_[0][j]<<" "<<D_[1][j]<<" "<<D_[2][j]<<" "<<D_[3][j]<<endl;
	//}



	int xdim = 4;


	for(unsigned int i1=0;i1<4;i1++)
	{
		for(unsigned int i2=0;i2<4;i2++)
		{
			track.M[i1][i2]=0.0;
			for(unsigned int j=0;j<2*n;j++)
			{
				track.M[i1][i2]+=track.D[i1][j]*track.D[i2][j];
			}
		}
	}

	invert(M,4);

	for(unsigned int j=0;j<2*n;j++) {
		for(unsigned int i1=0;i1<4;i1++) {
			track.MinvDt[i1][j]=0.0;
			for(unsigned int i2=0;i2<4;i2++) {
				track.MinvDt[i1][j]+=track.M[i1][i2+4]*track.D[i2][j];
			}
		}
	}

}





__device__
void linearTrackFit(GPUL1TTrack& track) {
	int numLayers = 6;

	int n = track.seed.nStubs;

	//Next calculate the residuals

	double delta[40];

	double chisq=0;

	unsigned int j=0;

	for(int i=0;i<n;i++) {
		double ri = r(track,i);
		double zi = track.seed.z[i];
		double phii=phi(track,i);
		double sigmax = track.seed.sigmax[i];
		double sigmaz = track.seed.sigmaz[i];

		int layer=track.seed.layer[i];

//		if (layer<1000) {
			//we are dealing with a barrel stub

			double deltaphi=track.phi0-asin(0.5*ri*track.rinv)-phii;
			if (deltaphi>CUDART_PI) phiprojapprox-=2*CUDART_PI;
			else
				if (deltaphi<-CUDART_PI) phiprojapprox+=2*CUDART_PI;

//			assert(fabs(deltaphi)<0.1*two_pi);

			delta[j++]=ri*deltaphi/sigmax;
			delta[j++]=(track.z0+(2.0/track.rinv)*t_*asin(ri*track.rinv/2)-zi)/sigmaz;


			//numerical derivative check

			for (int iii=0;iii<0;iii++){

				double drinv=0.0;
				double dphi0=0.0;
				double dt=0.0;
				double dz0=0.0;

				if (iii==0) drinv=0.001*fabs(track.rinv);
				if (iii==1) dphi0=0.001;
				if (iii==2) dt=0.001;
				if (iii==3) dz0=0.01;

				double deltaphi=phi0_+dphi0-asin(0.5*ri*(track.rinv+drinv))-phii;
				if (deltaphi>CUDART_PI) phiprojapprox-=2*CUDART_PI;
				else
					if (deltaphi<-CUDART_PI) phiprojapprox+=2*CUDART_PI;
//				assert(fabs(deltaphi)<0.1*two_pi);

				double delphi=ri*deltaphi/sigmax;
				double deltaz=(z0_+dz0+(2.0/(track.rinv+drinv))*(t_+dt)*asin(0.5*ri*(track.rinv+drinv))-zi)/sigmaz;


//				if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
//						<<(delphi-delta[j-2])/drinv<<" "
//						<<(deltaz-delta[j-1])/drinv<<endl;
//
//				if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
//						<<(delphi-delta[j-2])/dphi0<<" "
//						<<(deltaz-delta[j-1])/dphi0<<endl;
//
//				if (iii==2) cout << "Numerical t derivative: "<<i<<" "
//						<<(delphi-delta[j-2])/dt<<" "
//						<<(deltaz-delta[j-1])/dt<<endl;
//
//				if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
//						<<(delphi-delta[j-2])/dz0<<" "
//						<<(deltaz-delta[j-1])/dz0<<endl;

			}



//		}
//		else {
//			//we are dealing with a disk hit
//
//			double r_track=2.0*sin(0.5*rinv_*(zi-z0_)/t_)/rinv_;
//			//cout <<"t_track 1: "<<r_track<<endl;
//			double phi_track=phi0_-0.5*rinv_*(zi-z0_)/t_;
//
//			int iphi=stubs_[i].iphi();
//
//			double width=4.608;
//			double nstrip=508.0;
//			if (ri<60.0) {
//				width=4.8;
//				nstrip=480;
//			}
//			double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
//
//			if (stubs_[i].z()>0.0) Deltai=-Deltai;
//
//			double theta0=asin(Deltai/ri);
//
//			double Delta=Deltai-r_track*sin(theta0-(phi_track-phii));
//
//			delta[j++]=(r_track-ri)/sigmaz;
//			//double deltaphi=phi_track-phii;
//			//if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
//			//if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
//			//assert(fabs(deltaphi)<0.1*two_pi);
//			//delta[j++]=deltaphi/(sigmax/ri);
//			delta[j++]=Delta/sigmax;
//
//			//numerical derivative check
//
//			for (int iii=0;iii<0;iii++){
//
//				double drinv=0.0;
//				double dphi0=0.0;
//				double dt=0.0;
//				double dz0=0.0;
//
//				if (iii==0) drinv=0.001*fabs(rinv_);
//				if (iii==1) dphi0=0.001;
//				if (iii==2) dt=0.001;
//				if (iii==3) dz0=0.01;
//
//				r_track=2.0*sin(0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt))/(rinv_+drinv);
//				//cout <<"t_track 2: "<<r_track<<endl;
//				phi_track=phi0_+dphi0-0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt);
//
//				iphi=stubs_[i].iphi();
//
//				double width=4.608;
//				double nstrip=508.0;
//				if (ri<60.0) {
//					width=4.8;
//					nstrip=480;
//				}
//				Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
//
//				if (stubs_[i].z()>0.0) Deltai=-Deltai;
//				theta0=asin(Deltai/ri);
//
//				Delta=Deltai-r_track*sin(theta0-(phi_track-phii));
//
//				if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
//						<<((r_track-ri)/sigmaz-delta[j-2])/drinv<<" "
//						<<(Delta/sigmax-delta[j-1])/drinv<<endl;
//
//				if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
//						<<((r_track-ri)/sigmaz-delta[j-2])/dphi0<<" "
//						<<(Delta/sigmax-delta[j-1])/dphi0<<endl;
//
//				if (iii==2) cout << "Numerical t derivative: "<<i<<" "
//						<<((r_track-ri)/sigmaz-delta[j-2])/dt<<" "
//						<<(Delta/sigmax-delta[j-1])/dt<<endl;
//
//				if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
//						<<((r_track-ri)/sigmaz-delta[j-2])/dz0<<" "
//						<<(Delta/sigmax-delta[j-1])/dz0<<endl;
//
//			}
//
//		}

		chisq+=(delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1]);

	}

	double drinv=0.0;
	double dphi0=0.0;
	double dt=0.0;
	double dz0=0.0;

	double drinv_cov=0.0;
	double dphi0_cov=0.0;
	double dt_cov=0.0;
	double dz0_cov=0.0;



	for(unsigned int j=0;j<2*n;j++) {
		drinv-=MinvDt_[0][j]*delta[j];
		//cout << "MinvDt_[0][j] delta[j]:"<<MinvDt_[0][j]<<" "<<delta[j]<<endl;
		dphi0-=MinvDt_[1][j]*delta[j];
		dt-=MinvDt_[2][j]*delta[j];
		dz0-=MinvDt_[3][j]*delta[j];

		drinv_cov+=D_[0][j]*delta[j];
		dphi0_cov+=D_[1][j]*delta[j];
		dt_cov+=D_[2][j]*delta[j];
		dz0_cov+=D_[3][j]*delta[j];
	}


	double deltaChisq=drinv*drinv_cov+dphi0*dphi0_cov+dt*dt_cov+dz0*dz0_cov;

	//drinv=0.0; dphi0=0.0; dt=0.0; dz0=0.0;

	track.rinvfit=track.rinv+drinv;
	track.phi0fit=track.phi0+dphi0;

	track.tfit=t_+dt;
	track.z0fit=z0_+dz0;

	track.chisq1=(chisq+deltaChisq);
	track.chisq2=0.0;

	//cout << "Trackfit:"<<endl;
	//cout << "rinv_ drinv: "<<rinv_<<" "<<drinv<<endl;
	//cout << "phi0_ dphi0: "<<phi0_<<" "<<dphi0<<endl;
	//cout << "t_ dt      : "<<t_<<" "<<dt<<endl;
	//cout << "z0_ dz0    : "<<z0_<<" "<<dz0<<endl;

}


__device__
void residuals(GPUL1TTrack& track, double& largestresid,int& ilargestresid) {

	//   unsigned int n=stubs_.size();

	//Next calculate the residuals

	double delta[2];

//	double chisq=0.0;
	int n = track.seed.nStubs;

	largestresid=-1.0;
	ilargestresid=-1;

	for(int i=0;i<n;i++)
	{
		double ri = r(track,i);
		double zi = track.seed.z[i];

		double sigmax = track.seed.sigmax[i];
		double sigmaz = track.seed.sigmaz[i];

		double phii=stubs_[i].phi();
		double sigmax=stubs_[i].sigmax();
		double sigmaz=stubs_[i].sigmaz();

		int layer=stubs_[i].layer();

		//		if (layer<1000) {
		//we are dealing with a barrel stub

		double deltaphi=phi0fit_-asin(0.5*ri*rinvfit_)-phii;
		if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
		if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
		//		assert(fabs(deltaphi)<0.1*two_pi);

		delta[0]=ri*deltaphi/sigmax;
		delta[1]=(z0fit_+(2.0/rinvfit_)*tfit_*asin(0.5*ri*rinvfit_)-zi)/sigmaz;

//		}
//		else {
//			//we are dealing with a disk hit
//
//			double r_track=2.0*sin(0.5*rinvfit_*(zi-z0fit_)/tfit_)/rinvfit_;
//			double phi_track=phi0fit_-0.5*rinvfit_*(zi-z0fit_)/tfit_;
//
//			int iphi=stubs_[i].iphi();
//
//			double width=4.608;
//			double nstrip=508.0;
//			if (ri<60.0) {
//				width=4.8;
//				nstrip=480;
//			}
//			double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
//
//			if (stubs_[i].z()>0.0) Deltai=-Deltai;
//
//			double theta0=asin(Deltai/ri);
//
//			double Delta=Deltai-r_track*sin(theta0-(phi_track-phii));
//
//			delta[j++]=(r_track-ri)/sigmaz;
//			delta[j++]=Delta/sigmax;
//		}

		if (fabs(delta[0])>largestresid) {
			largestresid=fabs(delta[0]);
			ilargestresid=i;
		}

		if (fabs(delta[1])>largestresid) {
			largestresid=fabs(delta[1]);
			ilargestresid=i;
		}


//		chisq+=delta[0]*delta[0]+delta[1]*delta[1];

	}


}


__global__
void GPUTrackletFinding (GPUJobDescriptor* job )
{


	if(threadIdx.x == 0)
		nTrackletsInCell = 0;
	int numSectors = 24;
	int numLayers = 6;
	int sectorIdOrigin = blockIdx.x;
	int layerIdOrigin = blockIdx.y;


	int sectorIdTarget = (sectorIdOrigin + blockIdx.z - 1)% numSectors;
	int layerIdTarget = layerIdOrigin + 1;

	int indexCellOrigin = sectorIdOrigin + layerIdOrigin*numSectors;
	int indexCellTarget = sectorIdTarget + layerIdTarget*numSectors;

	int warpId = threadIdx.x/32; //warpId, each warp takes one element of stubs
	int threadInWarp = threadIdx.x % 32; // thread in warp
	if (warpId < job->numberOfElements[indexCellOrigin] && threadInWarp < job->StubsGPU[indexCellOrigin][warpId].nStubsInElement)
	{
		double r1 = r(&(job->StubsGPU[indexCellOrigin][warpId]), threadInWarp);
		double z1 = job->StubsGPU[indexCellOrigin][warpId].z[threadInWarp];
		double phi1 = phi(&(job->StubsGPU[indexCellOrigin][warpId]), threadInWarp);
		double sigmaz1 = job->StubsGPU[indexCellOrigin][warpId].sigmaz[threadInWarp];
		for( int indexElementTarget = 0; indexElementTarget < job->numberOfElements[indexCellTarget]; ++indexElementTarget )
		{
			for ( int indexStubTarget = 0; indexStubTarget < job->StubsGPU[indexCellTarget][indexElementTarget].nStubsInElement;
					++indexStubTarget)
			{

				double r2 = r(&(job->StubsGPU[indexCellTarget][indexElementTarget]), indexStubTarget);
				double z2 = job->StubsGPU[indexCellTarget][indexElementTarget].z[indexStubTarget];

				double zcrude=z1-(z2-z1)*r1/(r2-r1);
				if (fabs(zcrude)>30) continue;
				double phi2 = phi(&(job->StubsGPU[indexCellTarget][indexElementTarget]), indexStubTarget);
				double deltaphi = phi1-phi2;


				if(deltaphi > CUDART_PI)
					deltaphi -= 2*CUDART_PI;
				else
					if(deltaphi < -CUDART_PI)
						deltaphi+= 2*CUDART_PI;

				double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
				double rinv=2*sin(deltaphi)/dist;

				if (fabs(rinv)>0.0057) continue;

				double phi0=phi1+asin(0.5*r1*rinv);

				if (phi0>CUDART_PI) phi0-=2*CUDART_PI;
				if (phi0<-CUDART_PI) phi0+=2*CUDART_PI;

				double rhopsi1=2*asin(r1*rinv/2)/rinv;

				double rhopsi2=2*asin(r2*rinv/2)/rinv;

				double t=(z1-z2)/(rhopsi1-rhopsi2);

				double z0=z1-t*rhopsi1;

				if (sigmaz1>1.0) {
					if (fabs(z1-z2)<10.0){
						z0=0.0;
						t=z1/rhopsi1;
					}
				}

				if (fabs(z0)>15.0) continue;

				double pt1=	job->StubsGPU[indexCellOrigin][warpId].pt[threadInWarp];
				double pt2=job->StubsGPU[indexCellTarget][indexElementTarget].pt[indexStubTarget];
				//				double pttracklet=0.3*3.8/(rinv*100);
				double pttracklet=0.0114/rinv;
				bool pass1=2*fabs(1.0/pt1-1.0/pttracklet)<1.;
				bool pass2=2*fabs(1.0/pt2-1.0/pttracklet)<1.;
				bool pass=pass1&&pass2;

				if (!pass) continue;

				// Tracklet[idLayer*numSectors + idSector][i]
				// nTracklets[idLayer*numSectors + idSector][i]
				// increase the number of tracklets in the sector


				int oldNumTrackletsInCell = atomicAdd(&numTrackletsInCell[indexCellOrigin], 1);
				// modify the last-1
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].rinv = rinv;
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].phi0 = phi0;
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].t = t;
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].z0 = z0;


				// resetting track information from previous event
				for ( int sectorId = 0; sectorId < numSectors; ++sectorId)
					job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].StubInLayer[sectorId] = false;

				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].StubInLayer[layerIdOrigin] = true;
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].StubInLayer[layerIdTarget] = true;

// copying the stub in the tracklet information
// copying the origin stub first
// not copying simtrackid (needed?)
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].iphi[layerIdOrigin]  =  job->StubsGPU[indexCellOrigin][warpId].iphi[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].iz[layerIdOrigin]    =  job->StubsGPU[indexCellOrigin][warpId].iz[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].layer[layerIdOrigin] =  job->StubsGPU[indexCellOrigin][warpId].layer[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].ladder[layerIdOrigin]=  job->StubsGPU[indexCellOrigin][warpId].ladder[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].module[layerIdOrigin]=  job->StubsGPU[indexCellOrigin][warpId].module[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].x[layerIdOrigin]     =  job->StubsGPU[indexCellOrigin][warpId].x[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].y[layerIdOrigin]     =  job->StubsGPU[indexCellOrigin][warpId].y[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].z[layerIdOrigin]     =  job->StubsGPU[indexCellOrigin][warpId].z[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].sigmax[layerIdOrigin]=  job->StubsGPU[indexCellOrigin][warpId].sigmax[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].sigmaz[layerIdOrigin]=  job->StubsGPU[indexCellOrigin][warpId].sigmaz[threadInWarp];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].pt[layerIdOrigin]    =  job->StubsGPU[indexCellOrigin][warpId].pt[threadInWarp];

				// now copying the target stub
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].iphi[layerIdTarget]  =  job->StubsGPU[indexCellTarget][indexElementTarget].iphi[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].iz[layerIdTarget]    =  job->StubsGPU[indexCellTarget][indexElementTarget].iz[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].layer[layerIdTarget] =  job->StubsGPU[indexCellTarget][indexElementTarget].layer[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].ladder[layerIdTarget]=  job->StubsGPU[indexCellTarget][indexElementTarget].ladder[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].module[layerIdTarget]=  job->StubsGPU[indexCellTarget][indexElementTarget].module[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].x[layerIdTarget]     =  job->StubsGPU[indexCellTarget][indexElementTarget].x[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].y[layerIdTarget]     =  job->StubsGPU[indexCellTarget][indexElementTarget].y[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].z[layerIdTarget]     =  job->StubsGPU[indexCellTarget][indexElementTarget].z[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].sigmax[layerIdTarget]=  job->StubsGPU[indexCellTarget][indexElementTarget].sigmax[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].sigmaz[layerIdTarget]=  job->StubsGPU[indexCellTarget][indexElementTarget].sigmaz[indexStubTarget];
				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].pt[layerIdTarget]    =  job->StubsGPU[indexCellTarget][indexElementTarget].pt[indexStubTarget];

				job->intermediateData->Tracklet[indexCellOrigin][oldNumTrackletsInCell].nStubs               = 2;
			}
		}

	}

}


// NEED to build the following LUT
//
//findMatches(L[0],phiWindowSF_,0.04,0.5);
//findMatches(L[1],phiWindowSF_,0.025,3.0);
//findMatches(L[2],phiWindowSF_,0.075,0.5);
//
//findMatches(L[3],phiWindowSF_,0.075,3.0);
//findMatches(L[4],phiWindowSF_,0.1,3.0);
//findMatches(L[5],phiWindowSF_,0.15,3.0);
//
//




__global__
void GPUStubsMatching (GPUJobDescriptor* job )
{
// now we need to expand the tracklets by matching them to the stubs in the remaining cells
// the approach is a la gather: each thread takes an output element (tracklet) and cycles
// among the stubs in the other layers. These are then added to the tracklet information in case
// they're compatible with the tracklet.



	int numSectors = 24;
	int numLayers = 6;

	int sectorIdTracklet = blockIdx.x;
	int layerIdTracklet  = blockIdx.y;
	int cellIdTracklet   = layerIdTracklet*numSectors + sectorIdTracklet;

	int trackletIdInCell    = threadIdx.x;

	if(trackletIdInCell < nTrackletsInCell[trackletIdInCell])
	{

		double rinv= job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].rinv;
		double phi0= job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].phi0;
		double t   = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].t;
		double z0  = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].z0;


		for(int layerOffset = 0; layerOffset < numLayers-2 ; ++layerOffset )
		{
			int layerIdTarget = (layerIdTracklet + layerOffset + 2) % numLayers;
			// check with Anders if this is really needed
			double scale=1.0;
			double cutrphi = job->cutrphi[layerIdTarget]*scale;
			double cutrz = job->cutrz[layerIdTarget]*scale;
			double phiWindowSF = job->phiWindowSF*scale;

			uint32_t best_iphi  ;
			uint32_t best_iz    ;
			uint32_t best_layer ;
			uint32_t best_ladder;
			uint32_t best_module;
			double   best_x     ;
			double   best_y     ;
			double   best_z     ;
			double   best_sigmax;
			double   best_sigmaz;
			double   best_pt    ;

			double distbest=2e30;
			for(int sectorOffset = 0; sectorOffset < 3; ++sectorOffset)
			{

				int sectorIdTarget   = (sectorIdTracklet + sectorOffset - 1) % numSectors;
				int indexCellTarget = sectorIdTarget + layerIdTarget*numSectors;
				double phiprojapprox;
				double zprojapprox;
				double rapprox;
				if(job->numberOfElements[indexCellTarget] > 0)
				{
					rapprox=r(&(job->StubsGPU[indexCellTarget][0]), 0);

					phiprojapprox=phi0-asin(0.5*rapprox*rinv);
					zprojapprox=z0+2*t*asin(0.5*rapprox*rinv)/rinv;
					if (phiprojapprox>CUDART_PI) phiprojapprox-=2*CUDART_PI;
					if (phiprojapprox<-CUDART_PI) phiprojapprox+=2*CUDART_PI;
				}


				for( int indexElementTarget = 0; indexElementTarget < job->numberOfElements[indexCellTarget]; ++indexElementTarget )
				{
					for ( int indexStubTarget = 0; indexStubTarget < job->StubsGPU[indexCellTarget][indexElementTarget].nStubsInElement;
							++indexStubTarget)
					{
						double z = job->StubsGPU[indexCellTarget][indexElementTarget].z[indexStubTarget];
						if (fabs(z-zprojapprox)>10.0) continue;
						double phi = phi(&(job->StubsGPU[indexCellTarget][indexElementTarget]), indexStubTarget);
						double deltaphiapprox=fabs(phi-phiprojapprox);
						if (deltaphiapprox*rapprox>1.0) continue;
						double r=r(&(job->StubsGPU[indexCellTarget][indexElementTarget]), indexStubTarget);


						double phiproj=phi0-asin(0.5*r*rinv);
						double zproj=z0+2*t*asin(0.5*r*rinv)/rinv;
						double deltaphi=phi-phiproj;
						if (deltaphi>CUDART_PI) phiprojapprox-=2*CUDART_PI;
						else
							if (deltaphi<-CUDART_PI) phiprojapprox+=2*CUDART_PI;

						double rdeltaphi=r*deltaphi;
						double deltaz=z-zproj;

						if (fabs(rdeltaphi)>cutrphi*phiWindowSF) continue;
						if (fabs(deltaz)>cutrz) continue;

						double dist=hypot(rdeltaphi/cutrphi,deltaz/cutrz);

						if (dist<distbest)
						{
							distbest=dist;
							best_iphi    = job->StubsGPU[indexCellTarget][indexElementTarget].iphi[indexStubTarget];
							best_iz      = job->StubsGPU[indexCellTarget][indexElementTarget].iz[indexStubTarget];
							best_layer   = job->StubsGPU[indexCellTarget][indexElementTarget].layer[indexStubTarget];
							best_ladder  = job->StubsGPU[indexCellTarget][indexElementTarget].ladder[indexStubTarget];
							best_module  = job->StubsGPU[indexCellTarget][indexElementTarget].module[indexStubTarget];
							best_x       = job->StubsGPU[indexCellTarget][indexElementTarget].x[indexStubTarget];
							best_y       = job->StubsGPU[indexCellTarget][indexElementTarget].y[indexStubTarget];
							best_z       = job->StubsGPU[indexCellTarget][indexElementTarget].z[indexStubTarget];
							best_sigmax  = job->StubsGPU[indexCellTarget][indexElementTarget].sigmax[indexStubTarget];
							best_sigmaz  = job->StubsGPU[indexCellTarget][indexElementTarget].sigmaz[indexStubTarget];
							best_pt      = job->StubsGPU[indexCellTarget][indexElementTarget].pt[indexStubTarget];

						}
					}//end on loop inside one element
				}// end of loop inside array of elements
			}// end of loop sectors -1 0 1

			// now that we have fixed a layer, we have the best stub among the sectors of the tracklet -1 0 1
			if (distbest<1e30)
			{
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].iphi  [layerIdTarget]  = best_iphi;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].iz    [layerIdTarget]    = best_iz;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].layer [layerIdTarget] = best_layer;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].ladder[layerIdTarget]= best_ladder;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].module[layerIdTarget]= best_module;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].x     [layerIdTarget]     = best_x;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].y     [layerIdTarget]     = best_y;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].z     [layerIdTarget]     = best_z;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].sigmax[layerIdTarget]= best_sigmax;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].sigmaz[layerIdTarget]= best_sigmaz;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].pt    [layerIdTarget]    = best_pt;
				job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].nStubs++;



			}







		}// end of loop over all the layers


		// Now the each tracklet is matched to all the best  possible stubs.
		//
		// we need to fit the tracks now.
		// If a tracklet contains more than 2 stubs, it is possible to construct a track out of it.


		if(job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].nStubs > 3)
		{
			int oldNumTracksInSector = atomicAdd(&nTracksInSector[sectorIdTracklet], 1);
			int indexStubInTrack = 0;
			for ( int indexStub = 0; indexStub < numLayers; ++indexStub)
			{
				if (job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].StubInLayer[indexStub])
				{
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.iphi[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].iphi[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.iz[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].iz[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.layer[indexStubInTrack] = indexStub;
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.ladder[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].ladder[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.module[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].module[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.x[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].x[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.y[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].y[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.z[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].z[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.sigmax[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].sigmax[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.sigmaz[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].sigmaz[indexStub];
					job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.pt[indexStubInTrack] = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].pt[indexStub];

					indexStubInTrack++;






				}
			}
			double   phi0;
			double   z0;
			double   t;

			job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].rinv = job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].rinv;
			job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].phi0= job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].phi0;
			job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].z0= job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].z0;
			job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].t= job->intermediateData->Tracklet[cellIdTracklet][trackletIdInCell].t;
			job->intermediateData->Track[sectorIdTracklet][oldNumTracksInSector].seed.nStubs = indexStubInTrack;



		}


	}
}


__global__
void GPUTrackBuilding ( GPUJobDescriptor* job )
{

	int trackIdInSector = threadIdx.x + blockIdx.x * blockDim.x;

	int numSectors = 24;
	int sectorId = blockIdx.y;
	if(sectorId < numSectors && trackIdInSector < nTracksInSector[sectorId])
	{
	// 1 track per thread
	double largestResidual;
	int iLargestResidual;
	job->intermediateData->Track[sectorId][trackIdInSector].rinv=job->intermediateData->Track[sectorId][trackIdInSector].rinvfit;
	job->intermediateData->Track[sectorId][trackIdInSector].phi0=job->intermediateData->Track[sectorId][trackIdInSector].phi0fit;
	job->intermediateData->Track[sectorId][trackIdInSector].z0=job->intermediateData->Track[sectorId][trackIdInSector].z0fit;
	job->intermediateData->Track[sectorId][trackIdInSector].t=job->intermediateData->Track[sectorId][trackIdInSector].tfit;
	calculateDerivatives(job->intermediateData->Track[sectorId][trackIdInSector]);
	linearTrackFit(job->intermediateData->Track[sectorId][trackIdInSector]);

	largestresid=-1.0;
	ilargestresid=-1;

	residuals(job->intermediateData->Track[sectorId][trackIdInSector],largestresid,ilargestresid);
	//
	//    //cout << "Chisq largestresid: "<<chisq()<<" "<<largestresid<<endl;
	//
	//    if (stubs_.size()>3&&chisq()>100.0&&largestresid>5.0) {
	//      //cout << "Refitting track"<<endl;
	//      stubs_.erase(stubs_.begin()+ilargestresid);
	//      rinv_=rinvfit_;
	//      phi0_=phi0fit_;
	//      z0_=z0fit_;
	//      t_=tfit_;
	//      calculateDerivatives();
	//      linearTrackFit();
	//      residuals(largestresid,ilargestresid);
	//    }


	}

}

