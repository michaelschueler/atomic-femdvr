#---------------------------------------------------------------------------------
#   NOTES: 
#
#................................................
#          DIRECTORY STRUCTURE
#................................................
SRCDIR= Coulomb
MODDIR= Coulomb
F2PY= python3 -m numpy.f2py
#====================================================
# MAKE TARGETs: 
#        clean - remove all objects, modules and 
#	         binaries
#        prune - remove backup files created by
#	         emacs editor
#        info  - print basic information about the
#                host
#      serial  - make all serial programs 
#====================================================
.PHONY: clean prune purge info

coulomb:
	$(F2PY) $(SRCDIR)/COUL90.f -m CoulombWF -h $(MODDIR)/CoulombWF.pyf
	$(F2PY) -c $(MODDIR)/CoulombWF.pyf $(SRCDIR)/COUL90.f

clean:
	rm -rf $(MODDIR)/*.pyf
	rm -rf *.so


