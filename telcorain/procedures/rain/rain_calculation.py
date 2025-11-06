import traceback
from typing import Any

import numpy as np
from pycomlink.processing.baseline import baseline_constant
from pycomlink.processing.k_R_relation import calc_R_from_A
from pycomlink.processing.wet_antenna import (
    waa_schleiss_2013,
    waa_leijnse_2008_from_A_obs,
    waa_pastorek_2021_from_A_obs,
)
from xarray import Dataset

from telcorain.handlers.logging_handler import logger
from telcorain.procedures.exceptions import RaincalcException
from telcorain.procedures.rain import temperature_compensation, temperature_correlation
from telcorain.procedures.utils.external_filter import determine_wet
from telcorain.procedures.utils.helpers import measure_time

# import lib.pycomlink.pycomlink.processing as pycmlp
# from lib.pycomlink.pycomlink.processing.wet_dry import cnn
# from lib.pycomlink.pycomlink.processing.wet_dry.cnn import CNN_OUTPUT_LEFT_NANS_LENGTH

from telcorain.procedures.wet_dry import cnn
from telcorain.procedures.wet_dry.cnn import (
    CNN_OUTPUT_LEFT_NANS_LENGTH,
)  # CNN_OUTPUT_LEFT_NANS_LENGTH = 327

from telcorain.procedures.wet_dry import preprocess_utility

from telcorain.procedures.wet_dry.cnn_utility import (
    cnn_infer_only,
    attach_cnn_output_to_xarray,
)


@measure_time
def get_rain_rates(
    calc_data: list[Dataset],
    cp: dict[str, Any],
    ips: list[str],
    log_run_id: str = "default",
) -> list[Dataset]:
    current_link = 0

    try:
        logger.info("[%s] Smoothing signal data...", log_run_id)
        # link_count = len(calc_data)
        count = 0

        # links stored in this list will be removed from the calculation in case of enabled correlation filtering
        links_to_delete = []

        for link in calc_data:
            # TODO: load upper tx power from options (here it's 40 dBm)
            link["tsl"] = link.tsl.astype(float).where(link.tsl < 40.0)
            link["tsl"] = link.tsl.astype(float).interpolate_na(
                dim="time", method="nearest", max_gap=None
            )

            # TODO: load bottom rx power from options (here it's -70 dBm)
            link["rsl"] = (
                link.rsl.astype(float).where(link.rsl != 0.0).where(link.rsl > -70.0)
            )
            link["rsl"] = link.rsl.astype(float).interpolate_na(
                dim="time", method="nearest", max_gap=None
            )

            link["trsl"] = link.tsl - link.rsl

            link["temperature_rx"] = link.temperature_rx.astype(float).interpolate_na(
                dim="time", method="linear", max_gap=None
            )

            link["temperature_tx"] = link.temperature_tx.astype(float).interpolate_na(
                dim="time", method="linear", max_gap=None
            )
            current_link += 1
            count += 1

            """
            # temperature_correlation  - remove links if the correlation exceeds the specified threshold
            # temperature_compensation - as correlation, but also replaces the original trsl with the corrected one,
                                         according to the custom temperature compensation algorithm
            """

            if cp["temp"]["is_temp_filtered"]:
                logger.debug("[%s] Remove-link procedure started.", log_run_id)
                temperature_correlation.pearson_correlation(
                    count=count,
                    ips=ips,
                    curr_link=current_link,
                    link_todelete=links_to_delete,
                    link=link,
                    spin_correlation=cp["temp"]["correlation_threshold"],
                )

            if cp["temp"]["is_temp_compensated"]:
                logger.debug(
                    "[%s] Compensation algorithm procedure started.", log_run_id
                )
                temperature_compensation.compensation(
                    count=count,
                    ips=ips,
                    curr_link=current_link,
                    link=link,
                    spin_correlation=cp["temp"]["correlation_threshold"],
                )

            """
            'current_link += 1' serves to accurately list the 'count' and ip address of CML unit
             when the 'temperature_compensation.py' or 'temperature_correlation.py' is called
            """
            current_link += 1

        # Run the removal of high correlation links in case of enabled filtering
        for link in links_to_delete:
            calc_data.remove(link)

        # process each link -> get intensity R value for each link:
        logger.debug("[%s] Computing rain values...", log_run_id)
        current_link = 0

        if cp["wet_dry"]["is_mlp_enabled"]:
            if cp["wet_dry"]["cnn_model"] == "polz":
                for link in calc_data:
                    # determine wet periods using CNN
                    link["wet"] = (("time",), np.zeros([link.time.size]))

                    cnn_out = cnn.cnn_wet_dry(
                        trsl_channel_1=link.isel(channel_id=0).trsl.values,
                        trsl_channel_2=link.isel(channel_id=1).trsl.values,
                        threshold=0.82,
                        batch_size=128,
                    )

                    link["wet"] = (
                        ("time",),
                        np.where(np.isnan(cnn_out), link["wet"], cnn_out),
                    )
                # remove first CNN_OUTPUT_LEFT_NANS_LENGTH time values from dataset since they are NaNs
                calc_data = [
                    link.isel(time=slice(CNN_OUTPUT_LEFT_NANS_LENGTH, None))
                    for link in calc_data
                ]
            else:
                for i, link in enumerate(calc_data):
                    # ______________________________________________________________________________________________
                    # Debugging section
                    # it seems all trsl values are nan

                    # Check if TRSL values of link are NaN before preprocessing
                    if np.all(np.isnan(link.trsl.values)):
                        logger.warning(
                            "[%s] [CML ID: %d] All TRSL values are NaN before preprocessing",
                            log_run_id,
                            link.cml_id.values,
                        )

                    # ______________________________________________________________________________________________

                    preprocessed_df = preprocess_utility.cml_preprocess(
                        cml=link,
                        interp_max_gap=10,
                        suppress_step=True,
                        conv_threshold=250.0,
                        std_method=True,
                        window_size=10,
                        std_threshold=5.0,
                        z_method=True,
                        z_threshold=10.0,
                        reset_detect=False,
                        subtract_median=False,      # dont use for real time inference
                    )

                    
                    # ______________________________________________________________________________________________
                    # Debug section
                    # it seems all trsl values are nan after preprocessing -> check why

                    # Check if TRSL values of link are NaN 
                    if np.all(np.isnan(preprocessed_df.trsl_A.values)):
                        logger.warning(
                            "[%s] [CML ID: %d] All TRSL values are NaN after preprocessing",
                            log_run_id,
                            link.cml_id.values,
                        )
                    # ______________________________________________________________________________________________


                    cnn_out = cnn_infer_only(
                        preprocessed_df=preprocessed_df,
                        # param_dir="cnn_polz_ds_cz_param_2025-05-13_17;19",
                        # param_dir="cnn_v22_ds_cz_param_2025-05-15_22;01",
                        param_dir=cp["wet_dry"]["cnn_model_name"],
                        sample_size=60,
                    )
                    
                    link = attach_cnn_output_to_xarray(
                        link, cnn_out, sample_size=60, threshold=0.5
                    )
                    calc_data[i] = link

                logger.info(f"[{log_run_id}] Rain rates using custom CNN finished.")
        else:
            for link in calc_data:
                # determine wet periods using rolling standard deviation
                link["wet"] = (
                    link.trsl.rolling(
                        time=cp["wet_dry"]["rolling_values"],
                        center=cp["wet_dry"]["is_window_centered"],
                    ).std(skipna=False)
                    > cp["wet_dry"]["wet_dry_deviation"]
                )

        if cp["raingrids"]["is_external_filter_enabled"]:
            for link in calc_data:
                # central points of the links are sent into external filter
                link["lat_center"] = (link.site_a_latitude + link.site_b_latitude) / 2
                link["lon_center"] = (link.site_a_longitude + link.site_b_longitude) / 2

                for t in range(len(link.time)):
                    time = link.time[t].values
                    external_wet = determine_wet(
                        time,
                        link.lon_center,
                        link.lat_center,
                        cp["external_filter"]["radius"] + link.length / 2,
                        cp["external_filter"]["pixel_threshold"],
                        cp["external_filter"]["IMG_X_MIN"],
                        cp["external_filter"]["IMG_X_MAX"],
                        cp["external_filter"]["IMG_Y_MIN"],
                        cp["external_filter"]["IMG_Y_MAX"],
                        cp["external_filter"]["url"],
                        cp["external_filter"]["default_return"],
                        not cp["realtime"]["is_realtime"],
                    )
                    internal_wet = link.wet[t].values
                    link.wet[t] = external_wet and internal_wet
                    logger.debug(
                        "[%s] [EXTERNAL FILTER] CML: %d, time: %s, EXWET: %s && INTWET: %s = %s",
                        log_run_id,
                        link.cml_id.values,
                        time,
                        external_wet,
                        internal_wet,
                        link.wet[t].values,
                    )

        for link in calc_data:
            # calculate ratio of wet periods
            link["wet_fraction"] = (link.wet == 1).sum() / len(link.time)

            # determine signal baseline
            link["baseline"] = baseline_constant(
                trsl=link.trsl,
                wet=link.wet,
                n_average_last_dry=cp["wet_dry"]["baseline_samples"],
            )

            link["A_rain"] = link.trsl - link.baseline
            link["A_rain"].values[link.A_rain < 0] = 0

            if cp["waa"]["waa_method"] == "schleiss":
                # Schleiss WAA estimation
                link["waa"] = waa_schleiss_2013(
                    rsl=link.trsl,
                    baseline=link.baseline,
                    wet=link.wet,
                    waa_max=cp["waa"]["waa_schleiss_val"],
                    delta_t=60 / ((60 / cp["time"]["step"]) * 60),
                    tau=cp["waa"]["waa_schleiss_tau"],
                )
            elif cp["waa"]["waa_method"] == "leijnse":
                # Leijnse WAA estimation
                link["waa"] = waa_leijnse_2008_from_A_obs(
                    A_obs=link.A_rain,
                    f_Hz=link.frequency * 1e9,
                    pol=link.polarization,
                    L_km=float(link.length),
                )

            elif cp["waa"]["waa_method"] == "pastorek":
                # Pastorek WAA estimation
                link["waa"] = waa_pastorek_2021_from_A_obs(
                    A_obs=link.A_rain,
                    f_Hz=link.frequency * 1e9,
                    pol=link.polarization,
                    L_km=float(link.length),
                    A_max=2.2,
                )
            else:
                link["waa"] = link["A"]
                link["waa"] = link["waa"].where(link["waa"] >= 0, 0)

            # calculate final rain attenuation
            # link["A"] = link.trsl - link.baseline - link.waa # puvodni telcorain
            link["A"] = link.A_rain - link["waa"]
            link["A"] = link["A"].where(link["A"] >= 0, 0)

            # calculate rain intensity
            link["R"] = calc_R_from_A(
                A=link.A,
                L_km=float(link.length),
                f_GHz=link.frequency,
                pol=link.polarization,
            )

            current_link += 1
        return calc_data

    except BaseException as error:

        logger.error(
            "[%s] An unexpected error occurred during rain calculation: %s %s.\n"
            "Last processed microwave link dataset:\n%s\n"
            "Calculation thread terminated.",
            log_run_id,
            type(error),
            error,
            calc_data[current_link],
        )

        traceback.print_exc()

        raise RaincalcException("Error occurred during rainfall calculation processing")
