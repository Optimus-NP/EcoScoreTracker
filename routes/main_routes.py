from flask import Blueprint, render_template, request, jsonify, redirect, url_for
import logging

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@main_bp.route('/packaging')
def packaging():
    """Packaging suggestion page"""
    return render_template('packaging.html')

@main_bp.route('/carbon-footprint')
def carbon_footprint():
    """Carbon footprint prediction page"""
    return render_template('carbon_footprint.html')

@main_bp.route('/product-recommendation')
def product_recommendation():
    """Product recommendation page"""
    return render_template('product_recommendation.html')

@main_bp.route('/esg-score')
def esg_score():
    """ESG score prediction page"""
    return render_template('esg_score.html')

@main_bp.route('/batch-processing')
def batch_processing():
    """Batch processing page"""
    return render_template('batch_processing.html')

@main_bp.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@main_bp.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal error: {str(error)}")
    return render_template('500.html'), 500
