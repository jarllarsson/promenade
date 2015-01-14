# =======================================================================================================
# SETUP
# =======================================================================================================
out_w = 1280
out_h = 800
errstep = 1

set terminal unknown
plot 'CollectedRunsResultSerialBIPED.gnuplot.txt' using 1:2 # gather basic statistics
min_biped=GPVAL_DATA_Y_MIN
max_biped=GPVAL_DATA_Y_MAX

set terminal unknown
plot 'CollectedRunsResultSerialQUADRUPED.gnuplot.txt' using 1:2 # gather basic statistics
min_quadruped=GPVAL_DATA_Y_MIN
max_quadruped=GPVAL_DATA_Y_MAX

#default output
set terminal pngcairo size out_w,out_h enhanced font "Verdana,20"
set output "render/output_raster_ratios_serial.png"

# settings
set yrange [1:100]
set xrange [1:]
# set autoscale

set bars small

# define axis
# remove border on top and right and set color to gray
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
# define grid
set style line 12 lc rgb '#AFAFAF' lt 0 lw 1
set grid back ls 12

# color definitions
# lines
set style line 1 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 3 # --- green
set style line 2 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 3 # --- red
set style line 3 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 3 # --- blue
set style line 4 lc rgb '#FF5B00' pt -1 ps 1 lt 1 lw 3 # --- orange (for fit)
# error bars
set style line 11 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 0.3 # --- red
set style line 22 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 0.3 # --- green
set style line 33 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 0.3 # --- blue

set style fill transparent solid 0.2 noborder

set key off

set xlabel 'Character count'
set ylabel 'Ratio'

set label 2 'Biped Algorithm Slowdown Rate' at 30,18 left tc rgb "#4B632D"
set label 3 'Quadruped Algorithm Slowdown Rate' at 70,68 right tc rgb "#5c1720"

# =======================================================================================================
#
# PLOT ALL
#
# =======================================================================================================
#set autoscale
set ytics 5 font "Calibri,10"
set xtics add ("1" 1)
set xtics 5 font "Calibri,10"
set ytics add ("1" 1)
plot \
"CollectedRunsResultSerialBIPED.gnuplot.txt" using ($1+1):($2/min_biped) with lines ls 1 t 'Biped execution slowdown rate', \
"CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):($2/min_quadruped) with lines ls 2 t 'Quadruped execution slowdown rate'


# PDF
set terminal pdf enhanced font 'Calibri,10'
set output "render/output_vector_ratios_serial.pdf"
replot



# Live (wxWidgets)
set terminal wxt size out_w,out_h enhanced font 'Verdana,25' persist
replot
