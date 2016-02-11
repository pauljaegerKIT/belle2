/*
 * V0extra.cc
 *
 *  Created on: Jun 29, 2015
 *      Author: pjaeger
 */
#include <mdst/dataobjects/V0extra.h>

using namespace Belle2;

ClassImp(V0extra);

V0extra::V0extra(): V0()
{}

V0extra::V0extra(const std::pair<Belle2::Track*, Belle2::TrackFitResult*>& trackPairPositive,
                 const std::pair<Belle2::Track*, Belle2::TrackFitResult*>& trackPairNegative,
                 const TVector3& vertexPosition, const TMatrixDSym& covMatrix, const int& nTracks, const double& mReco)://, const int& pdg_value):
  V0(trackPairPositive, trackPairNegative), m_position(vertexPosition), cov_(covMatrix), nTracks(nTracks),
  mReco(mReco) //,pdg_value(pdg_value)
{}

double V0extra::getX() const
{
  double rad = m_position.X();
  return rad;
}

double V0extra::getY() const
{
  double rad = m_position.Y();
  return rad;
}

double V0extra::getXErr() const
{
  double xErr = sqrt(cov_[0][0]);
  return xErr;
}

double V0extra::getYErr() const
{
  double YErr = sqrt(cov_[1][1]);
  return YErr;
}

double V0extra::getMReco() const
{
  double YErr = mReco;
  return YErr;
}

int V0extra::getNTracks() const
{
  double n = nTracks;
  return n;
}
