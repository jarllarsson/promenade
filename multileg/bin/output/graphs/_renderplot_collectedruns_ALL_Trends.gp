# =======================================================================================================
# SETUP
# =======================================================================================================
out_w = 1280
out_h = 800
errstep = 1
#default output
set terminal pngcairo size out_w,out_h enhanced font "Verdana,20"
set output "render/collectedRuns_ALL_raster_trends.png"

# settings
set yrange [0:16]
set xrange [1:1000]

set bars small

# define axis
# remove border on top and right and set color to gray
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
# define grid
set style line 12 lc rgb '#808080' lt 0 lw 1
#set grid back ls 12

# color definitions
# lines
set style line 1 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 5 # --- green
set style line 2 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 3 # --- red
set style line 3 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 3 # --- blue
set style line 4 lc rgb '#FF5B00' pt -1 ps 1 lt 1 lw 3 # --- orange
set style line 5 lc rgb '#996633' pt -1 ps 1 lt 1 lw 3 # --- brown
set style line 6 lc rgb '#800080' pt -1 ps 1 lt 1 lw 3 # --- purple
set style line 7 lc rgb '#ADDFAD' pt -1 ps 1 lt 1 lw 3 # --- moss
set style line 8 lc rgb '#000000' pt -1 ps 1 lt 1 lw 3 # --- black
# trend lines
set style line 9 lc rgb '#000000' pt -1 ps 1 lt 1 lw 3 # --- black
set style line 10 lc rgb '#94CCA7' pt -1 ps 1 lt 1 lw 1.5 # --- light green
set style line 11 lc rgb '#911D4D' pt -1 ps 1 lt 0 lw 4 # --- red
# error bars
set style line 22 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 0.3 # --- red
set style line 33 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 0.3 # --- green
set style line 44 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 0.3 # --- blue

set style fill transparent solid 0.2 noborder

set key top left

set xlabel 'Character Count'
set ylabel 'Milliseconds'




linearfit(x) = m + k*x
m=0.0
fit linearfit(x) "CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):2 via k
polyfit(x) = a + b*x + c*x**2
fit polyfit(x) "CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):2 via a,b,c

# =======================================================================================================
#
# PLOT ALL
#
# =======================================================================================================
#set autoscale
set ytics 0.5 font "Calibri,15"
set xtics 100 font "Calibri,15"
set xtics add ("1" 1)
plot \
"CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):2 with lines ls 1 t 'Quadruped Serial', \
polyfit(x) with lines ls 11 t 'Polynomial Trend', \
linearfit(x) with lines ls 10 t 'Linear Trend'
#"CollectedRunsResultParallelQUADRUPED2.gnuplot.txt" using ($1+1):2 with lines ls 2 t 'Quadruped Parallel2', \
#"CollectedRunsResultParallelQUADRUPED3.gnuplot.txt" using ($1+1):2 with lines ls 3 t 'Quadruped Parallel3', \
#"CollectedRunsResultParallelQUADRUPED4.gnuplot.txt" using ($1+1):2 with lines ls 4 t 'Quadruped Parallel4', \
#"CollectedRunsResultSerialBIPED.gnuplot.txt" using ($1+1):2 with lines ls 5 t 'Biped Serial', \
#"CollectedRunsResultParallelBIPED2.gnuplot.txt" using ($1+1):2 with lines ls 6 t 'Biped Parallel2', \
#"CollectedRunsResultParallelBIPED3.gnuplot.txt" using ($1+1):2 with lines ls 7 t 'Biped Parallel3', \
#"CollectedRunsResultParallelBIPED4.gnuplot.txt" using ($1+1):2 with lines ls 8 t 'Biped Parallel4'




# PDF
set terminal pdf enhanced font 'Calibri,20'
set output "render/collectedRuns_ALL_vector_trends_zoomout.pdf"
replot

# PDF Zoom in
set yrange [0:4]
set xrange [1:250]

set arrow from 145,3.25 to 200,2.4 head filled lc rgb "#C69BAD"

set terminal pdf enhanced font 'Calibri,20'
set output "render/collectedRuns_ALL_vector_trends_zoomin.pdf"
replot

# PDF Zoom in
set autoscale
set yrange [0:]
set xrange [1:]


set terminal pdf enhanced font 'Calibri,20'
set output "render/collectedRuns_ALL_vector_trends_nozoom.pdf"
replot


# Live (wxWidgets)
set terminal wxt size out_w,out_h enhanced font 'Verdana,25' persist
replot

