#include "HeterogeneousHGCalHEFRecHitProducer.h"

HeterogeneousHGCalHEFRecHitProducer::HeterogeneousHGCalHEFRecHitProducer(const edm::ParameterSet& ps):
  token_(consumes<HGCUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("HGCHEFUncalibRecHitsTok")))
{
  //histogram drawing for debugging: to be removed at a later stage
  histo1_ = fs->make<TH1F>( "energy"  , "E", 100,  0., .2 );
  histo2_ = fs->make<TH1F>( "time"  , "t", 100,  0., 10. );
  histo3_ = fs->make<TH1F>( "timeError"  , "time_error", 100,  0., 10. );
  histo4_ = fs->make<TH1I>( "son"  , "son", 32,  0., 31. );

  nhitsmax_                 = ps.getParameter<uint32_t>("nhitsmax");
  cdata_.hgcHEF_keV2DIGI_   = ps.getParameter<double>("HGCHEF_keV2DIGI");
  cdata_.xmin_              = ps.getParameter<double>("minValSiPar"); //float
  cdata_.xmax_              = ps.getParameter<double>("maxValSiPar"); //float
  cdata_.aterm_             = ps.getParameter<double>("constSiPar"); //float
  cdata_.cterm_             = ps.getParameter<double>("noiseSiPar"); //float
  cdata_.rangeMatch_        = ps.getParameter<uint32_t>("rangeMatch");
  cdata_.rangeMask_         = ps.getParameter<uint32_t>("rangeMask");
  cdata_.hgcHEF_isSiFE_     = ps.getParameter<bool>("HGCHEF_isSiFE");
  vdata_.fCPerMIP           = ps.getParameter< std::vector<double> >("HGCHEF_fCPerMIP");
  vdata_.cce                = ps.getParameter<edm::ParameterSet>("HGCHEF_cce").getParameter<std::vector<double> >("values");
  vdata_.noise_fC           = ps.getParameter<edm::ParameterSet>("HGCHEF_noise_fC").getParameter<std::vector<double> >("values");
  vdata_.rcorr              = ps.getParameter< std::vector<double> >("rcorr");
  vdata_.weights            = ps.getParameter< std::vector<double> >("weights");
  cdata_.fhOffset_          = ps.getParameter<uint32_t>("offset"); //ddd_->layers(true);
  cdata_.s_hgcHEF_fCPerMIP_ = vdata_.fCPerMIP.size();
  cdata_.s_hgcHEF_cce_      = vdata_.cce.size();
  cdata_.s_hgcHEF_noise_fC_ = vdata_.noise_fC.size();
  cdata_.s_rcorr_           = vdata_.rcorr.size();
  cdata_.s_weights_ = vdata_.weights.size();
  cdata_.hgchefUncalib2GeV_ = 1e-6 / cdata_.hgcHEF_keV2DIGI_;
  vdata_.waferTypeL = {0, 1, 2};//ddd_->retWaferTypeL(); if depends on geometry the allocation is tricky!
  cdata_.s_waferTypeL_ = vdata_.waferTypeL.size();

  begin = std::chrono::steady_clock::now();

  tools_.reset(new hgcal::RecHitTools());
  stride_ = ( (nhitsmax_-1)/32 + 1 ) * 32; //align to warp boundary

  allocate_memory_();

  convert_constant_data_(h_kcdata_);

  produces<HGChefRecHitCollection>(collection_name_);
}

HeterogeneousHGCalHEFRecHitProducer::~HeterogeneousHGCalHEFRecHitProducer()
{
  delete kmdata_;
  delete h_kcdata_;
  delete d_kcdata_;
  delete old_soa_;
  delete d_oldhits_;
  delete d_newhits_;
  delete d_newhits_final_;
  delete h_newhits_;

  end = std::chrono::steady_clock::now();
  std::cout << "Time difference (heterogeneous) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;
}

void HeterogeneousHGCalHEFRecHitProducer::acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder w) {
  const cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w), ctxState_};
  set_geometry_(setup);
  event.getByToken(token_, handle_hef_);
  const auto &hits_hef = *handle_hef_;

  unsigned int nhits = hits_hef.size();
  std::cout << "HEF hits: " << nhits << std::endl;
  convert_collection_data_to_soa_(hits_hef, old_soa_, nhits);

  kmdata_ = new KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA>(nhitsmax_, stride_, old_soa_, d_oldhits_, d_newhits_, d_newhits_final_, h_newhits_);
  KernelManagerHGCalRecHit kernel_manager(kmdata_);
  kernel_manager.run_kernels(h_kcdata_, d_kcdata_);
  new_soa_ = kernel_manager.get_output();
  
  //print_to_histograms(kmdata_->h_out, histo1_, histo2_, histo3_, histo4_, nhits);
  
  rechits_ = std::make_unique< HGCRecHitCollection >();
  convert_soa_data_to_collection_(*rechits_, new_soa_, nhits);
}

void HeterogeneousHGCalHEFRecHitProducer::produce(edm::Event& event, const edm::EventSetup& setup)
{
  cms::cuda::ScopedContextProduce ctx{ctxState_}; //only for GPU to GPU producers
  event.put(std::move(rechits_), collection_name_);
}

void HeterogeneousHGCalHEFRecHitProducer::allocate_memory_()
{
  old_soa_ = new HGCUncalibratedRecHitSoA();
  d_oldhits_ = new HGCUncalibratedRecHitSoA();
  d_newhits_ = new HGCUncalibratedRecHitSoA();
  d_newhits_final_ = new HGCRecHitSoA();
  h_newhits_ = new HGCRecHitSoA();
  h_kcdata_ = new KernelConstantData<HGChefUncalibratedRecHitConstantData>(cdata_, vdata_);
  d_kcdata_ = new KernelConstantData<HGChefUncalibratedRecHitConstantData>(cdata_, vdata_);

  //_allocate pinned memory for constants on the host
  memory::allocation::host(h_kcdata_, h_mem_const_);
  //_allocate pinned memory for constants on the device
  memory::allocation::device(d_kcdata_, d_mem_const_);
  //_allocate memory for hits on the host
  memory::allocation::host(stride_, old_soa_, h_mem_in_);
  //_allocate memory for hits on the device
  memory::allocation::device(stride_, d_oldhits_, d_newhits_, d_newhits_final_, d_mem_);
  //_allocate memory for hits on the host
  memory::allocation::host(stride_, h_newhits_, h_mem_out_);
}

void HeterogeneousHGCalHEFRecHitProducer::set_geometry_(const edm::EventSetup& setup)
{
  tools_->getEventSetup(setup);
  std::string handle_str;
  handle_str = "HGCalHESiliconSensitive";
  edm::ESHandle<HGCalGeometry> handle;
  setup.get<IdealGeometryRecord>().get(handle_str, handle);
  ddd_ = &(handle->topology().dddConstants());
}

void HeterogeneousHGCalHEFRecHitProducer::convert_constant_data_(KernelConstantData<HGChefUncalibratedRecHitConstantData> *kcdata)
{
  for(int i=0; i<kcdata->data.s_hgcHEF_fCPerMIP_; ++i)
    kcdata->data.hgcHEF_fCPerMIP_[i] = kcdata->vdata.fCPerMIP[i];
  for(int i=0; i<kcdata->data.s_hgcHEF_cce_; ++i)
    kcdata->data.hgcHEF_cce_[i] = kcdata->vdata.cce[i];
  for(int i=0; i<kcdata->data.s_hgcHEF_noise_fC_; ++i)
    kcdata->data.hgcHEF_noise_fC_[i] = kcdata->vdata.noise_fC[i];
  for(int i=0; i<kcdata->data.s_rcorr_; ++i)
    kcdata->data.rcorr_[i] = kcdata->vdata.rcorr[i];
  for(int i=0; i<kcdata->data.s_weights_; ++i)
    kcdata->data.weights_[i] = kcdata->vdata.weights[i];
  for(int i=0; i<kcdata->data.s_waferTypeL_; ++i)
    kcdata->data.waferTypeL_[i] = kcdata->vdata.waferTypeL[i];
}

void HeterogeneousHGCalHEFRecHitProducer::convert_collection_data_to_soa_(const edm::SortedCollection<HGCUncalibratedRecHit>& hits, HGCUncalibratedRecHitSoA* d, const unsigned int& nhits)
{
  for(unsigned int i=0; i<nhits; ++i)
    {
      d->amplitude[i] = hits[i].amplitude();
      d->pedestal[i] = hits[i].pedestal();
      d->jitter[i] = hits[i].jitter();
      d->chi2[i] = hits[i].chi2();
      d->OOTamplitude[i] = hits[i].outOfTimeEnergy();
      d->OOTchi2[i] = hits[i].outOfTimeChi2();
      d->flags[i] = hits[i].flags();
      d->aux[i] = 0;
      d->id[i] = hits[i].id().rawId();
    }
}

void HeterogeneousHGCalHEFRecHitProducer::convert_soa_data_to_collection_(HGCRecHitCollection& rechits, HGCRecHitSoA *d, const unsigned int& nhits)
{
  rechits.reserve(nhits);
  for(uint i=0; i<nhits; ++i)
    {
      DetId id_converted( d->id[i] );
      rechits.emplace_back( HGCRecHit(id_converted, d->energy[i], d->time[i], 0, d->flagBits[i]) );
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HeterogeneousHGCalHEFRecHitProducer);
