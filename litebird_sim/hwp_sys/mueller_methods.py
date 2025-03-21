
# -*- encoding: utf-8 -*-
import numpy as np
from numba import njit, prange

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""
- - - - - - METHODS FOR MUELLER TOD COMPUTATIONS - - - - - 
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""


@njit(parallel=False)
def compute_Tterm_for_one_sample_for_tod(mII, mQI, mUI, cos2Xi2Phi, sin2Xi2Phi):
    Tterm = mII + mQI * cos2Xi2Phi + mUI * sin2Xi2Phi

    return Tterm


@njit(parallel=False)
def compute_Qterm_for_one_sample_for_tod(
    mIQ, mQQ, mUU, mIU, mUQ, mQU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
):
    Qterm = np.cos(2 * psi + 2 * phi) * (
        mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi
    ) - np.sin(2 * psi + 2 * phi) * (mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi)

    return Qterm


@njit(parallel=False)
def compute_Uterm_for_one_sample_for_tod(
    mIU, mQU, mUQ, mIQ, mQQ, mUU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
):
    Uterm = np.sin(2 * psi + 2 * phi) * (
        mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi
    ) + np.cos(2 * psi + 2 * phi) * (mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi)

    return Uterm

################################################################
#################### SINGLE FREQUENCY ##########################
################################################################

@njit(parallel=False)
def compute_signal_for_one_sample(
    T,
    Q,
    U,
    mII,
    mQI,
    mUI,
    mIQ,
    mIU,
    mQQ,
    mUU,
    mUQ,
    mQU,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    """Bolometric equation, tod filling for a single (time) sample"""
    d = T * compute_Tterm_for_one_sample_for_tod(mII, mQI, mUI, cos2Xi2Phi, sin2Xi2Phi)

    d += Q * compute_Qterm_for_one_sample_for_tod(
        mIQ, mQQ, mUU, mIU, mUQ, mQU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )

    d += U * compute_Uterm_for_one_sample_for_tod(
        mIU, mQU, mUQ, mIQ, mQQ, mUU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )

    return d


@njit(parallel=False)
def compute_signal_for_one_detector(
    tod_det, pixel_ind, m0f, m2f, m4f, theta, psi, maps, cos2Xi2Phi, sin2Xi2Phi, phi
):
    """
    Single-frequency case: compute the signal for a single detector,
    looping over (time) samples.
    """

    for i in prange(len(tod_det)):
        FourRhoPsiPhi = 4 * (theta[i] - psi[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - psi[i] - phi)
        tod_det[i] += compute_signal_for_one_sample(
            T=maps[0, pixel_ind[i]],
            Q=maps[1, pixel_ind[i]],
            U=maps[2, pixel_ind[i]],
            mII=m0f[0, 0]
            + m2f[0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f[0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQI=m0f[1, 0]
            + m2f[1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f[1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUI=m0f[2, 0]
            + m2f[2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f[2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQ=m0f[0, 1]
            + m2f[0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f[0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIU=m0f[0, 2]
            + m2f[0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f[0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQ=m0f[1, 1]
            + m2f[1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f[1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUU=m0f[2, 2]
            + m2f[2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f[2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQ=m0f[2, 1]
            + m2f[2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f[2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQU=m0f[1, 2]
            + m2f[1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f[1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            psi=psi[i],
            phi=phi,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )


@njit(parallel=False)
def compute_TQUsolver_for_one_sample(
    mIIs,
    mQIs,
    mUIs,
    mIQs,
    mIUs,
    mQQs,
    mUUs,
    mUQs,
    mQUs,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Single-frequency case: computes :math:`A^T A` and :math:`A^T d`
    for a single detector, for one (time) sample.
    """

    Tterm = compute_Tterm_for_one_sample_for_tod(
        mIIs, mQIs, mUIs, cos2Xi2Phi, sin2Xi2Phi
    )

    Qterm = compute_Qterm_for_one_sample_for_tod(
        mIQs, mQQs, mUUs, mIUs, mUQs, mQUs, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )
    Uterm = compute_Uterm_for_one_sample_for_tod(
        mIUs, mQUs, mUQs, mIQs, mQQs, mUUs, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )

    return Tterm, Qterm, Uterm


@njit(parallel=False)
def compute_ata_atd_for_one_detector(
    ata,
    atd,
    tod,
    m0f_solver,
    m2f_solver,
    m4f_solver,
    pixel_ind,
    theta,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Single-frequency case: compute :math:`A^T A` and :math:`A^T d`
    for a single detector, looping over (time) samples.
    """

    for i in prange(len(tod)):
        FourRhoPsiPhi = 4 * (theta[i] - psi[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - psi[i] - phi)
        # psi_i = 2*np.arctan2(np.sqrt(quats_rot[i][0]**2 + quats_rot[i][1]**2 + quats_rot[i][2]**2),quats_rot[i][3])
        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=m0f_solver[0, 0]
            + m2f_solver[0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f_solver[0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQIs=m0f_solver[1, 0]
            + m2f_solver[1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f_solver[1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUIs=m0f_solver[2, 0]
            + m2f_solver[2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f_solver[2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQs=m0f_solver[0, 1]
            + m2f_solver[0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f_solver[0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIUs=m0f_solver[0, 2]
            + m2f_solver[0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f_solver[0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQs=m0f_solver[1, 1]
            + m2f_solver[1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f_solver[1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUUs=m0f_solver[2, 2]
            + m2f_solver[2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f_solver[2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQs=m0f_solver[2, 1]
            + m2f_solver[2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f_solver[2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQUs=m0f_solver[1, 2]
            + m2f_solver[1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f_solver[1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            psi=psi[i],
            phi=phi,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

        atd[pixel_ind[i], 0] += tod[i] * Tterm
        atd[pixel_ind[i], 1] += tod[i] * Qterm
        atd[pixel_ind[i], 2] += tod[i] * Uterm

        ata[pixel_ind[i], 0, 0] += Tterm * Tterm
        ata[pixel_ind[i], 1, 0] += Tterm * Qterm
        ata[pixel_ind[i], 2, 0] += Tterm * Uterm
        ata[pixel_ind[i], 1, 1] += Qterm * Qterm
        ata[pixel_ind[i], 2, 1] += Qterm * Uterm
        ata[pixel_ind[i], 2, 2] += Uterm * Uterm


################################################################
#################### BAND INTEGRATION ##########################
################################################################

@njit(parallel=False)
def integrate_inband_signal_for_one_sample(
    T,
    Q,
    U,
    freqs,
    band,
    mII,
    mQI,
    mUI,
    mIQ,
    mIU,
    mQQ,
    mUU,
    mUQ,
    mQU,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Multi-frequency case: band integration with trapezoidal rule,
    :math:`\sum (f(i) + f(i+1)) \cdot (\nu_(i+1) - \nu_i)/2`
    for a single (time) sample.
    """
    tod = 0
    for i in range(len(band) - 1):
        dnu = freqs[i + 1] - freqs[i]
        tod += (
            (
                band[i]
                * compute_signal_for_one_sample(
                    T=T[i],
                    Q=Q[i],
                    U=U[i],
                    mII=mII[i],
                    mQI=mQI[i],
                    mUI=mUI[i],
                    mIQ=mIQ[i],
                    mIU=mIU[i],
                    mQQ=mQQ[i],
                    mUU=mUU[i],
                    mUQ=mUQ[i],
                    mQU=mQU[i],
                    cos2Xi2Phi=cos2Xi2Phi,
                    sin2Xi2Phi=sin2Xi2Phi,
                )
                + band[i + 1]
                * compute_signal_for_one_sample(
                    T=T[i + 1],
                    Q=Q[i + 1],
                    U=U[i + 1],
                    mII=mII[i + 1],
                    mQI=mQI[i + 1],
                    mUI=mUI[i + 1],
                    mIQ=mIQ[i + 1],
                    mIU=mIU[i + 1],
                    mQQ=mQQ[i + 1],
                    mUU=mUU[i + 1],
                    mUQ=mUQ[i + 1],
                    mQU=mQU[i + 1],
                    cos2Xi2Phi=cos2Xi2Phi,
                    sin2Xi2Phi=sin2Xi2Phi,
                )
            )
            * dnu
            / 2
        )

    return tod

@njit(parallel=False)
def integrate_inband_signal_for_one_detector(
    tod_det, freqs, band, m0f,m2f,m4f, pixel_ind, theta, psi, maps, phi, cos2Xi2Phi, sin2Xi2Phi,
):
    """
    Multi-frequency case: band integration of the signal for a single detector,
    looping over (time) samples.
    """
    for i in range(len(tod_det)):
        FourRhoPsiPhi = 4 * (theta[i] - psi[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - psi[i] - phi)

        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=maps[:, 0, pixel_ind[i]],
            Q=maps[:, 1, pixel_ind[i]],
            U=maps[:, 2, pixel_ind[i]],
            freqs=freqs,
            band=band,
            mII=m0f[:,0, 0]
            + m2f[:,0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f[:,0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQI=m0f[:,1, 0]
            + m2f[:,1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f[:,1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUI=m0f[:,2, 0]
            + m2f[:,2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f[:,2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQ=m0f[:,0, 1]
            + m2f[:,0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f[:,0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIU=m0f[:,0, 2]
            + m2f[:,0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f[:,0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQ=m0f[:,1, 1]
            + m2f[:,1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f[:,1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUU=m0f[:,2, 2]
            + m2f[:,2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f[:,2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQ=m0f[:,2, 1]
            + m2f[:,2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f[:,2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQU=m0f[:,1, 2]
            + m2f[:,1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f[:,1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

@njit(parallel=False)
def integrate_inband_TQUsolver_for_one_sample(
    freqs,
    band,
    mIIs,
    mQIs,
    mUIs,
    mIQs,
    mIUs,
    mQQs,
    mUUs,
    mUQs,
    mQUs,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Multi-frequency case: band integration with trapezoidal rule,
    :math:`\sum (f(i) + f(i+1)) \cdot (\nu_(i+1) - \nu_i)/2`
    for a single (time) sample.
    """
    intTterm = 0
    intQterm = 0
    intUterm = 0
    for i in range(len(band) - 1):
        dnu = freqs[i + 1] - freqs[i]

        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=mIIs[i],
            mQIs=mQIs[i],
            mUIs=mUIs[i],
            mIQs=mIQs[i],
            mIUs=mIUs[i],
            mQQs=mQQs[i],
            mUUs=mUUs[i],
            mUQs=mUQs[i],
            mQUs=mQUs[i],
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

        Ttermp1, Qtermp1, Utermp1 = compute_TQUsolver_for_one_sample(
            mIIs=mIIs[i + 1],
            mQIs=mQIs[i + 1],
            mUIs=mUIs[i + 1],
            mIQs=mIQs[i + 1],
            mIUs=mIUs[i + 1],
            mQQs=mQQs[i + 1],
            mUUs=mUUs[i + 1],
            mUQs=mUQs[i + 1],
            mQUs=mQUs[i + 1],
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

        intTterm += (band[i] * Tterm + band[i + 1] * Ttermp1) * dnu / 2.0
        intQterm += (band[i] * Qterm + band[i + 1] * Qtermp1) * dnu / 2.0
        intUterm += (band[i] * Uterm + band[i + 1] * Utermp1) * dnu / 2.0

    return intTterm, intQterm, intUterm

@njit(parallel=False)
def integrate_inband_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
    freqs,
    band,
    m0f_solver,
    m2f_solver,
    m4f_solver,
    pixel_ind,
    theta,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Multi-frequency case: band integration of :math:`A^T A` and :math:`A^T d`
    for a single detector, looping over (time) samples.
    """
    for i in range(len(tod)):
        FourRhoPsiPhi = 4 * (theta[i] - psi[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - psi[i] - phi)
        Tterm, Qterm, Uterm = integrate_inband_TQUsolver_for_one_sample(
            freqs=freqs,
            band=band,
            mIIs=m0f_solver[:,0, 0]
            + m2f_solver[:,0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f_solver[:,0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQIs=m0f_solver[:,1, 0]
            + m2f_solver[:,1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f_solver[:,1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUIs=m0f_solver[:,2, 0]
            + m2f_solver[:,2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f_solver[:,2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQs=m0f_solver[:,0, 1]
            + m2f_solver[:,0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f_solver[:,0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIUs=m0f_solver[:,0, 2]
            + m2f_solver[:,0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f_solver[:,0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQs=m0f_solver[:,1, 1]
            + m2f_solver[:,1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f_solver[:,1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUUs=m0f_solver[:,2, 2]
            + m2f_solver[:,2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f_solver[:,2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQs=m0f_solver[:,2, 1]
            + m2f_solver[:,2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f_solver[:,2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQUs=m0f_solver[:,1, 2]
            + m2f_solver[:,1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f_solver[:,1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )
        atd[pixel_ind[i], 0] += tod[i] * Tterm
        atd[pixel_ind[i], 1] += tod[i] * Qterm
        atd[pixel_ind[i], 2] += tod[i] * Uterm

        ata[pixel_ind[i], 0, 0] += Tterm * Tterm
        ata[pixel_ind[i], 1, 0] += Tterm * Qterm
        ata[pixel_ind[i], 2, 0] += Tterm * Uterm
        ata[pixel_ind[i], 1, 1] += Qterm * Qterm
        ata[pixel_ind[i], 2, 1] += Qterm * Uterm
        ata[pixel_ind[i], 2, 2] += Uterm * Uterm