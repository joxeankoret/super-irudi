# Super irudi

Super Irudi is a command line based tool which aims to help automating some common tasks for enhancing photographs.

# Example usages

For example, let's say that you have one photograph with nice colors and another one with not that good colors and you want to 'copy' the color histogram from one to another, one can run a command line like the following one:
```$ ./super_irudi.py m74-bad-core.jpg -i -mh m51-good-core.jpg
[Sat May 14 16:10:46 2022] Matching histograms...
[Sat May 14 16:10:47 2022] Source image's size (4014, 3865)
[Sat May 14 16:10:49 2022] Reference's image size (11553, 7633)
[Sat May 14 16:11:01 2022] Displaying final image```

It will show the following preview:
![M74 as M51](https://github.com/joxeankoret/super-irudi/blob/main/examples/m74-as-m51.png?raw=true)
