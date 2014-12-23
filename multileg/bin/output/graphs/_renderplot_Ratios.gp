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
set style line 12 lc rgb '#808080' lt 0 lw 1
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

set key top left

set xlabel 'Character count'
set ylabel 'Ratio'


# =======================================================================================================
#
# PLOT ALL
#
# =======================================================================================================
#set autoscale
set ytics 5 font "Verdana,12" 
set xtics add ("1" 1)
set xtics 5 font "Verdana,12" 
set ytics add ("1" 1)
plot \
"CollectedRunsResultSerialBIPED.gnuplot.txt" using ($1+1):($2/min_biped) with lines ls 1 t 'Biped execution time ratio', \
"CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):($2/min_quadruped) with lines ls 2 t 'Quadruped execution time ratio'
#"perf_serial.gnuplot.txt" every errstep with yerrorbars ls 11 t '', "perf_serial.gnuplot.txt" with lines ls 1 t 'Serial', \
# "antsdat.txt" every errstep with yerrorbars ls 22 t '', "antsdat.txt" with lines ls 2 t 'Ants', \
# "spongedat.txt" every errstep with yerrorbars ls 33 t '', "spongedat.txt" with lines ls 3 t 'Sponge'

# EPS
#set terminal postscript size out_w,out_h eps enhanced color
#set output "render/output_vector_ratios_serial.eps"
#replot

# PDF
set terminal pdf
set output "render/output_vector_ratios_serial.pdf"
replot

# SVG
set terminal svg size out_w,out_h fname "Verdana" fsize 45
set output "render/output_vector_ratios_serial.svg"
replot

# Live (wxWidgets)
set terminal wxt size out_w,out_h enhanced font 'Verdana,25' persist
replot


# =======================================================================================================
#
# SEPARATE PLOTS
#
# =======================================================================================================
#   unset label
#   set xlabel 'Step'
#   set ylabel 'Population'
#   # =======================================================================================================
#   # WORMS
#   # =======================================================================================================
#   set terminal pngcairo size out_w,out_h/1.5 enhanced font "Verdana,45"
#   set yrange [-10:300]
#   set output "render/worms_output_raster.png"
#   # fit a function to the data
#   wormfit(x) = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5
#   fit wormfit(x) "wormsdat.txt" via a,b,c,d,e,f
#   # plot
#   plot wormfit(x) with lines ls 4 t 'Trend', \
#   "wormsdat.txt" every errstep with yerrorbars ls 11 t '', "wormsdat.txt" with lines ls 1 t 'Worms'
#   # EPS
#   set terminal postscript size out_w,out_h eps enhanced color
#   set output "render/worms_output_vector.eps"
#   replot
#   # SVG
#   set terminal svg size out_w,out_h fname "Verdana" fsize 45
#   set output "render/worms_output_vector.svg"
#   replot
#   
#   # =======================================================================================================
#   # ANTS
#   # =======================================================================================================
#   set terminal pngcairo size out_w,out_h/1.5 enhanced font "Verdana,45"
#   set yrange [-10:1800]
#   set output "render/ants_output_raster.png"
#   # fit a function to the data
#   antfit(x) = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5
#   fit antfit(x) "antsdat.txt" via a,b,c,d,e,f
#   # plot
#   plot antfit(x) with lines ls 4 t 'Trend', \
#   "antsdat.txt" every errstep with yerrorbars ls 22 t '', "antsdat.txt" with lines ls 2 t 'Ants'
#   # EPS
#   set terminal postscript size out_w,out_h eps enhanced color
#   set output "render/ants_output_vector.eps"
#   replot
#   # SVG
#   set terminal svg size out_w,out_h fname "Verdana" fsize 45
#   set output "render/ants_output_vector.svg"
#   replot
#   
#   # =======================================================================================================
#   # SPONGE
#   # =======================================================================================================
#   set terminal pngcairo size out_w,out_h/1.5 enhanced font "Verdana,45"
#   set yrange [-10:8600]
#   set output "render/sponge_output_raster.png"
#   # fit a function to the data
#   spongefit(x) = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5
#   fit spongefit(x) "spongedat.txt" via a,b,c,d,e,f
#   # plot
#   plot spongefit(x) with lines ls 4 t 'Trend', \
#   "spongedat.txt" every errstep with yerrorbars ls 33 t '', "spongedat.txt" with lines ls 3 t 'Sponge'
#   # EPS
#   set terminal postscript size out_w,out_h eps enhanced color
#   set output "render/sponge_output_vector.eps"
#   replot
#   # SVG
#   set terminal svg size out_w,out_h fname "Verdana" fsize 45
#   set output "render/sponge_output_vector.svg"
#   replot