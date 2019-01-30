void GenEvent(){


	TFile *f1 = new TFile("background.root","recreate");
	//TFile *f1 = new TFile("signal.root","recreate");
	TTree *mytree = new TTree("tree","tree");
	
  	double Feature1;
  	double Feature2;
  	double Feature3;
	int   isSig;

	mytree-> Branch("Feature1",&Feature1);
	mytree-> Branch("Feature2",&Feature2);
	mytree-> Branch("Feature3",&Feature3);
	mytree-> Branch("isSig",&isSig);

	for(int i=0; i<5000; i++){
	
// --Background
		Feature1 = gRandom->Gaus(1,3);
		Feature2 = gRandom->Gaus(10,2);
		Feature3 = gRandom->Poisson(7);
		isSig = 0;
		
// -- Signal
		//Feature1 = gRandom->Gaus(7,3);
		//Feature2 = gRandom->Gaus(1,2);
		//Feature3 = gRandom->Poisson(1);
		//isSig = 1;
		

	mytree->Fill();
	}

	f1->Write();




}
