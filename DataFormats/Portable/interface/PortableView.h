// #ifndef DataFormats_Portable_interface_PortableView_h
// #define DataFormats_Portable_interface_PortableView_h

// // #include <utility> 
// // // #include <string>
// // // #include <vector>
// #include "DataFormats/Portable/interface/PortableCollection.h"
// #include "DataFormats/SoATemplate/interface/SoAView.h"

// template <typename T>
// class PortableView {
// public:
//     using Layout = T;
//     using View = typename Layout::View;
//     using Buffer = cms::alpakatools::host_buffer<std::byte[]>;

//     PortableView() = default;

//     PortableView(Argument<Collections>&&... args) {}

//     ~PortableView() = default;

// private:
//     int32_t elements_;

// };

// #endif // DataFormats_Portable_interface_PortableView_h