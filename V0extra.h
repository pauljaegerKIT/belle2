/*
 * V0extra.h
 *
 * Dataobject contains the RaveFit information of Kshorts at reconstruciton level.
 * Information usually does not get saved, but for evaluation reasons it is compared 
 * to the information at Analysis level. Efficiencies and Vertex resolutions will be determined.
 * 
 * 
 * 
 *  Created on: Jun 29, 2015
 *      Author: pjaeger
 */

#pragma once
#include <mdst/dataobjects/V0.h>
#include <framework/datastore/RelationsObject.h>
#include <framework/datastore/StoreArray.h>
#include <TVector3.h>

namespace Belle2 {

  class V0extra : public V0 {
  public:
    /** Constructor without arguments; needed for I/O. */
    V0extra();

    /** Constructor taking two pairs of tracks and trackFitResults. */
    V0extra(const std::pair<Belle2::Track*, Belle2::TrackFitResult*>& trackPairPositive,
            const std::pair<Belle2::Track*, Belle2::TrackFitResult*>& trackPairNegative,
            const TVector3& vertexPosition, const TMatrixDSym& covMatrix, const int& nTracks, const double& mReco);
    double getX() const;
    double getY() const;
    double getXErr() const;
    double getYErr() const;
    int getNTracks() const;
    double getMReco() const;

  private:
    TVector3 m_position;
    TMatrixDSym cov_;
    int nTracks;
    double mReco;
    /** Macro for ROOTification. */
    ClassDef(V0extra, 1);


  };
}
